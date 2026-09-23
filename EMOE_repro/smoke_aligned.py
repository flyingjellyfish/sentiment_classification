"""Small GPU feasibility run of the authors' EMOE model on E-question aligned data.

This intentionally keeps the paper's regression task and core loss. It uses the
precomputed 768-D text feature and a narrower flattened-input router so that a
single 16 GB RTX 5060 Ti can train it. It is not a full benchmark reproduction.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EMOE_source"
sys.path.insert(0, str(SOURCE))

from trains.singleTask.model.emoe import EMOE  # noqa: E402
from trains.utils.functions import entropy_balance, uni_distill  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "E题数据" / "附件2-数据集特征文件" / "aligned_50.pkl",
    )
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--valid-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--router-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def model_args(router_hidden: int) -> argparse.Namespace:
    with (SOURCE / "config" / "config.json").open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    model_config = config["emoe"]
    merged = {
        **config["datasetCommonParams"]["mosei"]["aligned"],
        **model_config["commonParams"],
        **model_config["datasetParams"]["mosei"],
        "dataset_name": "mosei",
        "use_bert": False,
        "use_finetune": False,
        "router_hidden_dim": router_hidden,
    }
    return argparse.Namespace(**merged)


def make_subset(block: dict, limit: int, seed: int) -> TensorDataset:
    count = len(block["regression_labels"])
    if limit <= 0 or limit > count:
        raise ValueError(f"Requested {limit} samples from split of {count}")
    indices = np.random.default_rng(seed).choice(count, size=limit, replace=False)
    # The supplied pickle stores audio/vision in float64; cast only selected
    # samples to float32 to avoid multiplying host-memory use on Windows.
    text = torch.from_numpy(np.ascontiguousarray(block["text"][indices], dtype=np.float32))
    audio = torch.from_numpy(np.nan_to_num(block["audio"][indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    vision = torch.from_numpy(np.nan_to_num(block["vision"][indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    labels = torch.from_numpy(np.asarray(block["regression_labels"][indices], dtype=np.float32).reshape(-1, 1))
    expected = ((limit, 50, 768), (limit, 50, 74), (limit, 50, 35), (limit, 1))
    shapes = (tuple(text.shape), tuple(audio.shape), tuple(vision.shape), tuple(labels.shape))
    if shapes != expected:
        raise ValueError(f"Unexpected tensor shapes: {shapes}, expected {expected}")
    return TensorDataset(text, audio, vision, labels)


def original_emoe_loss(output: dict, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
    # Matches trains/singleTask/EMOE.py, with vectorized importance weights.
    # Losses stay float32 under AMP for numerical stability.
    pred = {key: output[key].float() for key in ("logits_c", "logits_l", "logits_v", "logits_a")}
    weights = output["channel_weight"].float()
    loss_multi = torch.nn.functional.l1_loss(pred["logits_c"], labels)
    loss_uni = sum(
        torch.nn.functional.l1_loss(pred[key], labels)
        for key in ("logits_l", "logits_v", "logits_a")
    ) / 3.0

    errors = torch.cat(
        [
            (pred["logits_l"] - labels).square(),
            (pred["logits_v"] - labels).square(),
            (pred["logits_a"] - labels).square(),
        ],
        dim=1,
    )
    inverse_errors = 1.0 / (errors + 0.1)
    importance = inverse_errors / inverse_errors.sum(dim=1, keepdim=True)
    loss_sim = (importance.detach() - weights).square().mean()
    loss_entropy = entropy_balance(weights)

    unimodal_projection = (
        output["l_proj"].float() * weights[:, 0:1]
        + output["v_proj"].float() * weights[:, 1:2]
        + output["a_proj"].float() * weights[:, 2:3]
    ).detach()
    loss_ud = uni_distill(output["c_proj"].float(), unimodal_projection)
    loss = loss_multi + loss_uni + 0.1 * (loss_entropy + 0.1 * loss_sim) + 0.1 * loss_ud
    parts = {
        "multi_mae": float(loss_multi.detach()),
        "uni_mae": float(loss_uni.detach()),
        "router_entropy": float(loss_entropy.detach()),
        "router_similarity": float(loss_sim.detach()),
        "distillation": float(loss_ud.detach()),
    }
    return loss, parts


@torch.no_grad()
def evaluate(model: EMOE, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    predictions, truths, weights = [], [], []
    for text, audio, vision, labels in loader:
        output = model(text.to(device), audio.to(device), vision.to(device))
        predictions.append(output["logits_c"].float().cpu().numpy().ravel())
        truths.append(labels.numpy().ravel())
        weights.append(output["channel_weight"].float().cpu().numpy())
    pred = np.concatenate(predictions)
    true = np.concatenate(truths)
    weight = np.concatenate(weights)
    if not np.isfinite(pred).all() or not np.isfinite(weight).all():
        raise RuntimeError("Non-finite model output during validation")
    corr = float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) > 0 and np.std(true) > 0 else None
    return {
        "mae": float(np.mean(np.abs(pred - true))),
        "pearson": corr,
        "mean_router_weights_LVA": weight.mean(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RTX 5060 Ti feasibility run")
    if torch.cuda.get_device_capability(0) != (12, 0):
        print("Notice: GPU capability differs from the expected RTX 5060 Ti sm_120")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.data.open("rb") as stream:
        data = pickle.load(stream)
    train_set = make_subset(data["train"], args.train_samples, args.seed)
    valid_set = make_subset(data["valid"], args.valid_samples, args.seed + 1)
    del data
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EMOE(model_args(args.router_hidden)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    torch.cuda.reset_peak_memory_stats(device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        last_parts = {}
        for text, audio, vision, labels in train_loader:
            text, audio, vision, labels = (x.to(device) for x in (text, audio, vision, labels))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model(text, audio, vision)
            loss, last_parts = original_emoe_loss(output, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation = evaluate(model, valid_loader, device)
        result = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
            "last_batch_loss_parts": last_parts,
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        history.append(result)

    report = {
        "source_commit": "c4759c748105e18354cd9082b1d8c6d310f1f9d8",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "data": str(args.data),
        "train_samples": args.train_samples,
        "valid_samples": args.valid_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "router_hidden": args.router_hidden,
        "use_bert": False,
        "amp": amp_enabled,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
        "history": history,
    }
    report_path = args.output_dir / "smoke_report.json"
    checkpoint_path = args.output_dir / "smoke_checkpoint.pt"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    torch.save(model.cpu().state_dict(), checkpoint_path)
    print(json.dumps({"report": str(report_path), "checkpoint": str(checkpoint_path), "peak_mib": report["peak_cuda_reserved_mib"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

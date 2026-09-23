"""Fine-tune the explicit missing-mask model from the augmented EMOE weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from q2_aligned import read_data, train_one, evaluate, scenarios, infer_special


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--router-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "q2" / "refined")
    parser.add_argument("--init", type=Path, default=Path(__file__).resolve().parent / "results" / "q2" / "augment.pt")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = read_data()
    model, valid_loader, run = train_one("full", data, device, args, args.output_dir, init_path=args.init)
    impact = {name: evaluate(model, valid_loader, device, "full", scenario) for name, scenario in scenarios()}
    output = infer_special(model, device, args.output_dir)
    report = {"initialization": str(args.init), "epochs": args.epochs, "lr": args.lr,
              "training": run, "impact": impact, "attachment3_predictions": str(output)}
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "predictions": str(output), "best_epoch": run["best_epoch"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()

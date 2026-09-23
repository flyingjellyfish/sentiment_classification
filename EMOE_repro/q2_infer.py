"""Run all 30 aligned attachment-3 samples from a saved Q2 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from q2_aligned import Q2EMOE, infer_special
from smoke_aligned import model_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "results" / "q2" / "refined" / "full.pt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "q2" / "final")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Q2EMOE(model_args(checkpoint["router_hidden"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = infer_special(model, device, args.output_dir)
    print(json.dumps({"checkpoint": str(args.checkpoint), "output": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

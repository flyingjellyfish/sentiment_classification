"""Compare unchanged deployed Q2 checkpoint with the staged warm control."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from q2_aligned import Q2EMOE, read_data, scenarios
from q2_staged_refine import DEFAULT_OUT, HERE, evaluate, make_model
from smoke_aligned import model_args


class OriginalAdapter(torch.nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = Q2EMOE(model_args(int(saved["router_hidden"]))).to(device)
        self.model.load_state_dict(saved["model"], strict=True)

    def forward(self, text, audio, vision, observed_mask=None, valid_lengths=None):
        return self.model(text, audio, vision, observed_mask=observed_mask)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid = read_data()["valid"]
    paths = {
        "current_refined": HERE / "results/q2/refined/full.pt",
        "warm_control": DEFAULT_OUT / "control/seed_1111/best.pt",
    }
    models = {"current_refined": OriginalAdapter(paths["current_refined"], device).eval(),
              "warm_control": make_model(paths["warm_control"], "control", device).eval()}
    rows = []
    for candidate, model in models.items():
        for name, scenario in scenarios():
            metric = evaluate(model, valid, device, scenario)
            rows.append({"candidate": candidate, "scenario": name,
                         **{key: metric[key] for key in ("accuracy", "f1_macro", "mae", "pearson")},
                         "actual_new_missing_fraction": metric.get("actual_new_missing_fraction", 0)})
    with (DEFAULT_OUT / "current_vs_warm_grid.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoints": {k: str(v.resolve()) for k, v in paths.items()},
               "n_valid": len(valid), "scenarios": len(scenarios()), "blocks": {}}
    for candidate in models:
        block = [r for r in rows if r["candidate"] == candidate]
        summary["blocks"][candidate] = {
            "clean": next(r for r in block if r["scenario"] == "clean"),
            "av_middle30": next(r for r in block if r["scenario"] == "audio_vision_middle_30pct"),
            "missing_grid_mean": {key: float(np.mean([r[key] for r in block if r["scenario"] != "clean"]))
                                  for key in ("accuracy", "f1_macro", "mae", "pearson")}}
    (DEFAULT_OUT / "current_vs_warm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

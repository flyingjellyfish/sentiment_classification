"""Static validation comparison plots from the completed paired Q2 runs."""
from __future__ import annotations

import csv
import os

from q2_dual_baseline import OUT


os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    with (OUT / "scenario_mean.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {(r["model"], r["scenario"]): r for r in csv.DictReader(stream)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=170)
    colors = {"emoe": "#1f77b4", "prmf_e": "#d62728"}
    for column, kind in enumerate(("text", "all_three")):
        for model in ("emoe", "prmf_e"):
            scenarios = ["clean"] + [f"{kind}_middle_{rate}pct" for rate in (10, 30, 50)]
            x = [0, 10, 30, 50]
            for row_index, (metric, label) in enumerate((("f1_macro", "Macro F1"), ("mae", "MAE"))):
                y = [float(rows[model, name][metric + "_mean"]) for name in scenarios]
                sd = [float(rows[model, name][metric + "_sd"]) for name in scenarios]
                axes[row_index, column].errorbar(x, y, yerr=sd, marker="o", capsize=3,
                                                 color=colors[model], label=model.upper())
                axes[row_index, column].set_xlabel("Nominal contiguous missing rate (%)")
                axes[row_index, column].set_ylabel(label)
                axes[row_index, column].grid(alpha=.25)
                axes[row_index, column].set_title(f"{kind.replace('_', ' ').title()} middle span")
    axes[0, 0].legend()
    fig.suptitle("E Q2 aligned_50 validation, 3-seed mean ± SD")
    fig.tight_layout()
    fig.savefig(OUT / "missing_rate_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=170)
    for i, (kind, metric, label) in enumerate((("text", "f1_macro", "Macro F1"),
                                             ("all_three", "f1_macro", "Macro F1"))):
        positions = ("start", "middle", "end")
        for j, model in enumerate(("emoe", "prmf_e")):
            values = [float(rows[model, f"{kind}_{p}_30pct"][metric + "_mean"]) for p in positions]
            axes[i].plot(range(3), values, marker="o", color=colors[model], label=model.upper())
        axes[i].set_xticks(range(3), positions)
        axes[i].set_ylabel(label)
        axes[i].set_title(f"{kind.replace('_', ' ').title()} 30% by position")
        axes[i].grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "missing_position_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(str(OUT / "missing_rate_curves.png"))


if __name__ == "__main__":
    main()

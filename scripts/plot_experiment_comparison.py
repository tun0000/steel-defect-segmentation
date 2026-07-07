#!/usr/bin/env python3
"""Plot per-class mask mAP comparison between two evaluate.py reports.

Parses the markdown tables in two ``eval_results.md`` files produced by
``scripts/evaluate.py`` (so every plotted number comes from a real evaluation
artifact) and renders grouped bar charts — mask mAP50-95 and mask mAP50 side
by side — with the per-class delta annotated above each pair.

Example:
    uv run python scripts/plot_experiment_comparison.py \\
        --baseline reports/eval_results.md \\
        --experiment reports/defect1_weighted/eval_results.md \\
        --experiment-label "defect_1-weighted" --highlight defect_1 \\
        --out reports/figures/defect1_weighted_comparison.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_eval_report(path: Path) -> dict[str, tuple[float, float]]:
    """Return {class_name: (mask mAP50, mask mAP50-95)} plus an 'overall' entry."""
    text = path.read_text()
    overall = re.search(r"mask mAP50: ([\d.]+), mask mAP50-95: ([\d.]+)", text)
    if overall is None:
        raise ValueError(f"no overall mask mAP line found in {path}")

    results: dict[str, tuple[float, float]] = {}
    header_cols: list[str] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header_cols is None:
            if "mask mAP50" in cells and "mask mAP50-95" in cells:
                header_cols = cells
            continue
        if set(cells[0]) <= {"-", ":", " "}:  # separator row
            continue
        map50 = float(cells[header_cols.index("mask mAP50")])
        map50_95 = float(cells[header_cols.index("mask mAP50-95")])
        results[cells[0]] = (map50, map50_95)
    if not results:
        raise ValueError(f"no per-class table found in {path}")
    results["overall"] = (float(overall.group(1)), float(overall.group(2)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline eval_results.md")
    parser.add_argument("--experiment", type=Path, required=True, help="experiment eval_results.md")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--experiment-label", default="experiment")
    parser.add_argument("--highlight", default=None, help="class name to shade as the experiment's target")
    parser.add_argument("--out", type=Path, required=True, help="output image path")
    args = parser.parse_args()

    base = parse_eval_report(args.baseline)
    exp = parse_eval_report(args.experiment)
    if set(base) != set(exp):
        raise ValueError(f"class mismatch between reports: {sorted(base)} vs {sorted(exp)}")
    categories = [c for c in base if c != "overall"] + ["overall"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    panels = [("mask mAP50-95 (strict IoU)", 1), ("mask mAP50", 0)]
    width = 0.36
    for ax, (panel_title, idx) in zip(axes, panels):
        base_vals = [base[c][idx] for c in categories]
        exp_vals = [exp[c][idx] for c in categories]
        xs = range(len(categories))
        top = max(*base_vals, *exp_vals) * 1.3
        if args.highlight in categories:
            ax.axvspan(categories.index(args.highlight) - 0.5, categories.index(args.highlight) + 0.5,
                       color="gray", alpha=0.12, zorder=0)
        ax.bar([x - width / 2 for x in xs], base_vals, width, label=args.baseline_label, color="#4c72b0")
        ax.bar([x + width / 2 for x in xs], exp_vals, width, label=args.experiment_label, color="#dd8452")
        for x, (b, e) in enumerate(zip(base_vals, exp_vals)):
            ax.text(x - width / 2, b + top * 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=7.5)
            ax.text(x + width / 2, e + top * 0.01, f"{e:.3f}", ha="center", va="bottom", fontsize=7.5)
            delta = e - b
            ax.text(x, max(b, e) + top * 0.06, f"{delta:+.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#2e7d32" if delta >= 0 else "#c62828")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(categories)
        if args.highlight in categories:
            ax.get_xticklabels()[categories.index(args.highlight)].set_fontweight("bold")
        ax.set_ylim(0, top)
        ax.set_title(panel_title)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mask mAP (val)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(f"{args.experiment_label} vs {args.baseline_label} — held-out validation mask mAP", y=0.98)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved chart to {args.out}")


if __name__ == "__main__":
    main()

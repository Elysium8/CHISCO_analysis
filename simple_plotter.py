#!/usr/bin/env python3
"""
plot_jsonl_metrics.py

Minimal JSONL training-curve plotter.

Each line in the JSONL should be a JSON object that includes an epoch field and
some metric fields (e.g., loss, val.loss, val.acc10, etc).

Examples:
  # One plot per metric (default)
  python plot_jsonl_metrics.py runs/a/metrics.jsonl runs/b/metrics.jsonl \
    --metrics val.acc10 val.loss

  # All metrics on one plot
  python plot_jsonl_metrics.py runs/a/metrics.jsonl runs/b/metrics.jsonl \
    --metrics val.acc10 val.loss --single-plot

  # Nested keys via dot notation
  python plot_jsonl_metrics.py /mnt/data/metrics.jsonl \
    --metrics train.loss val.loss --epoch-key epoch

Notes:
- If a metric appears multiple times per epoch, the last one in the file is used.
- Missing metrics are skipped with a warning.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def get_by_dotted_key(d: Dict[str, Any], key: str) -> Any:
    """Access nested dict keys using dot notation, e.g. 'val.loss'."""
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def read_jsonl_series(
    path: str,
    epoch_key: str,
    metrics: List[str],
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns dict: metric -> list of (epoch, value) sorted by epoch.
    Uses the last seen value per epoch for each metric.
    """
    per_metric_per_epoch: Dict[str, Dict[float, float]] = {m: {} for m in metrics}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch_val = get_by_dotted_key(obj, epoch_key)
            if epoch_val is None:
                continue
            try:
                epoch = float(epoch_val)
            except (TypeError, ValueError):
                continue

            for m in metrics:
                v = get_by_dotted_key(obj, m)
                if v is None:
                    continue
                try:
                    per_metric_per_epoch[m][epoch] = float(v)
                except (TypeError, ValueError):
                    continue

    out: Dict[str, List[Tuple[float, float]]] = {}
    for m in metrics:
        items = sorted(per_metric_per_epoch[m].items(), key=lambda t: t[0])
        out[m] = items
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="One or more JSONL metric file paths.")
    ap.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Metric keys to plot (supports dot notation). Example: val.acc10 val.loss",
    )
    ap.add_argument(
        "--epoch-key",
        default="epoch",
        help="Key for epoch (supports dot notation). Default: epoch",
    )
    ap.add_argument(
        "--single-plot",
        action="store_true",
        help="If set, plot all specified metrics on a single graph instead of one per metric.",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to save the figure (e.g., curves.png). If omitted, shows window.",
    )
    args = ap.parse_args()

    files = args.files
    metrics = args.metrics
    epoch_key = args.epoch_key

    if args.single_plot:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        axes = [ax]  # keep a uniform interface
    else:
        n = len(metrics)
        fig, axes = plt.subplots(n, 1, figsize=(9, 3.5 * n), squeeze=False)
        axes = axes[:, 0]

    any_plotted = False

    for path in files:
        run_label = os.path.splitext(os.path.basename(path))[0]
        series = read_jsonl_series(path, epoch_key=epoch_key, metrics=metrics)

        if args.single_plot:
            ax = axes[0]
            for m in metrics:
                pts = series.get(m, [])
                if not pts:
                    print(f"[warn] {run_label}: metric '{m}' not found (or no numeric values).")
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, label=f"{run_label}:{m}")
                ax.set_xlabel("epoch")
                ax.grid(True, alpha=0.3)
                any_plotted = True
        else:
            for ax, m in zip(axes, metrics):
                pts = series.get(m, [])
                if not pts:
                    print(f"[warn] {run_label}: metric '{m}' not found (or no numeric values).")
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, label=run_label)
                ax.set_xlabel("epoch")
                ax.set_ylabel(m)
                ax.grid(True, alpha=0.3)
                any_plotted = True

    if args.single_plot:
        axes[0].set_title("Metrics vs Epoch" if not args.title else args.title)
        axes[0].legend()
    else:
        for ax, m in zip(axes, metrics):
            ax.set_title(m)
            ax.legend()
        if args.title:
            fig.suptitle(args.title)

    fig.tight_layout()

    if not any_plotted:
        print("No data was plotted. Check --epoch-key and --metrics names against your JSONL.")
        return

    if args.out:
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

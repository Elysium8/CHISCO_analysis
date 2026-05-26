#!/usr/bin/env python3
"""
Plot averaged retrieval metrics vs train_data_fraction for experiment folders.

Folder structure:
  <experiments_root>/
    exp_001/
      config.json
      global_retrieval.json
    exp_002/
      config.json
      global_retrieval.json
    ...

Produces plots in <experiments_root>/:
  - Combined (default):
      metrics_vs_fraction_task_imagine.png
      metrics_vs_fraction_task_read.png
  - Split (--split-graphs):
      metrics_vs_fraction_task_imagine_top5.png, ...
      metrics_vs_fraction_task_read_top5.png, ...

Config must include at least:
  - "task": "read" or "imagine"
  - "train_data_fraction": float
(see example)

Eval JSON contains numeric metrics, including keys like top1/top5/... and "total".
We exclude "total". We also never plot top1. (see example)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

CONFIG_NAME = "config.json"
EVAL_NAME = "global_retrieval.json"

# Total candidate pool size for chance baseline
TOTAL_CANDIDATES = 6567


def safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def metric_sort_key(name: str) -> Tuple[int, str]:
    # If name contains digits (e.g., top1/top5), sort by numeric part first.
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        return (int(digits), name)
    return (10**9, name)


def is_top1_metric(metric_name: str) -> bool:
    """
    Return True if this metric is a top-1 style metric (e.g., "top1", "top_1", "top-1").
    """
    m = metric_name.strip().lower()
    if not m.startswith("top"):
        return False
    digits = "".join(ch for ch in m if ch.isdigit())
    return digits == "1"


def chance_for_metric(metric_name: str) -> float | None:
    """
    Return chance accuracy as a *percentage* for topK-style metrics:
      chance_percent = 100 * K / TOTAL_CANDIDATES
    If metric_name doesn't contain a numeric K (e.g., "mrr"), return None.
    """
    digits = "".join(ch for ch in metric_name if ch.isdigit())
    if not digits:
        return None
    k = int(digits)
    if k <= 0:
        return None
    return 100.0 * k / float(TOTAL_CANDIDATES)


def to_relative_accuracy(metric_name: str, value: float) -> float:
    """
    Convert an accuracy metric to relative accuracy (accuracy / chance),
    where chance is expressed as a percentage (so value should also be in percent).
    If chance can't be computed, return value unchanged.
    """
    chance = chance_for_metric(metric_name)
    if chance is None or chance == 0.0:
        return value
    return value / chance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, help="Folder containing experiment subfolders")
    ap.add_argument(
        "--plot-mode",
        choices=["absolute", "relative"],
        default="relative",
        help="Plot absolute accuracy (raw metric values) or relative accuracy (accuracy / chance). Default: relative.",
    )
    ap.add_argument(
        "--split-graphs",
        action="store_true",
        help="If set, produce a separate graph for each metric (per task). Default: combined.",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively (in addition to saving)",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root folder does not exist: {root}")

    # Save plots in the same folder as the experiments (the root folder)
    outdir = root

    # grouped[(task, frac)][metric] -> list of values
    grouped: Dict[Tuple[str, float], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_metric_names = set()

    exp_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not exp_dirs:
        raise SystemExit(f"No subfolders found under: {root}")

    skipped = 0
    used = 0

    for exp_dir in sorted(exp_dirs):
        cfg_path = exp_dir / CONFIG_NAME
        eval_path = exp_dir / EVAL_NAME

        if not cfg_path.is_file() or not eval_path.is_file():
            skipped += 1
            continue

        try:
            cfg = safe_read_json(cfg_path)
            ev = safe_read_json(eval_path)
        except Exception as e:
            print(f"[WARN] Could not read JSON in {exp_dir}: {e}")
            skipped += 1
            continue

        task = cfg.get("task", None)
        frac = cfg.get("train_data_fraction", None)

        if not isinstance(task, str) or not is_number(frac):
            print(f"[WARN] Missing/invalid task or train_data_fraction in {cfg_path}")
            skipped += 1
            continue

        task = task.strip().lower()
        frac_f = float(frac)

        # Metrics: numeric keys excluding "total" and excluding top1 entirely
        for k, v in ev.items():
            if k == "total":
                continue
            if is_top1_metric(k):
                continue
            if is_number(v):
                grouped[(task, frac_f)][k].append(float(v))
                all_metric_names.add(k)

        used += 1

    if used == 0:
        raise SystemExit(
            f"Found 0 usable experiments under {root}. "
            f"Expected each experiment folder to contain {CONFIG_NAME} and {EVAL_NAME}."
        )

    print(f"Ingested experiments: {used}")
    print(f"Skipped folders (missing files / invalid JSON / missing fields): {skipped}")

    metrics_sorted = sorted(all_metric_names, key=metric_sort_key)

    def y_transform(metric_name: str, avg_val: float) -> float:
        if args.plot_mode == "relative":
            return to_relative_accuracy(metric_name, avg_val)
        return avg_val

    def y_label(metric_name: str | None = None) -> str:
        if args.plot_mode == "relative":
            return "relative accuracy (accuracy / chance)"
        # absolute
        return metric_name if metric_name else "accuracy"

    def fig_suffix() -> str:
        return "relative" if args.plot_mode == "relative" else "absolute"

    def save_or_show(fig_name: str) -> None:
        outpath = outdir / fig_name
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        print(f"Wrote: {outpath}")
        if args.show:
            plt.show()
        else:
            plt.close()

    # -------- NEW: compute global y-limits across BOTH tasks (so axes match) --------
    global_y_min = float("inf")
    global_y_max = float("-inf")

    for task in ["imagine", "read"]:
        fracs = sorted({frac for (t, frac) in grouped.keys() if t == task})
        for metric in metrics_sorted:
            for frac in fracs:
                vals = grouped.get((task, frac), {}).get(metric, [])
                if not vals:
                    continue
                avg_val = sum(vals) / len(vals)
                y = y_transform(metric, avg_val)
                global_y_min = min(global_y_min, y)
                global_y_max = max(global_y_max, y)

    if global_y_min == float("inf") or global_y_max == float("-inf"):
        raise SystemExit("No plottable metric values found after filtering (e.g., top1 removal).")

    # optional padding to avoid touching plot border
    pad = 0.05 * (global_y_max - global_y_min) if global_y_max > global_y_min else 1.0
    ylim_low = global_y_min - pad
    ylim_high = global_y_max + pad
    # ------------------------------------------------------------------------------

    # Plot per task
    for task in ["imagine", "read"]:
        fracs = sorted({frac for (t, frac) in grouped.keys() if t == task})
        if not fracs:
            print(f"[INFO] No experiments for task='{task}', skipping.")
            continue

        if not args.split_graphs:
            # Combined plot per task
            plt.figure()
            for metric in metrics_sorted:
                xs, ys = [], []
                for frac in fracs:
                    vals = grouped.get((task, frac), {}).get(metric, [])
                    if vals:
                        xs.append(frac)
                        avg_val = sum(vals) / len(vals)
                        ys.append(y_transform(metric, avg_val))
                if xs:
                    plt.plot(xs, ys, marker="o", label=metric)

            plt.xlabel("Fraction of training data used", fontsize=14)
            plt.ylabel(y_label(), fontsize=14)
            # plt.title(f"Global retrieval metrics vs train_data_fraction (task={task}, mode={args.plot_mode})")
            plt.grid(True, linewidth=0.5, alpha=0.5)
            plt.legend()

            # -------- NEW: lock y-axis to be the same across tasks --------
            plt.ylim(ylim_low, ylim_high)
            # -------------------------------------------------------------

            save_or_show(f"metrics_vs_fraction_task_{task}_{fig_suffix()}.png")

        else:
            # Split: separate plot per metric per task
            for metric in metrics_sorted:
                xs, ys = [], []
                for frac in fracs:
                    vals = grouped.get((task, frac), {}).get(metric, [])
                    if vals:
                        xs.append(frac)
                        avg_val = sum(vals) / len(vals)
                        ys.append(y_transform(metric, avg_val))

                if not xs:
                    continue

                plt.figure()
                plt.plot(xs, ys, marker="o")
                plt.xlabel("Fraction of training data used")
                plt.ylabel(y_label(metric))
                plt.title(f"{metric} vs train_data_fraction (task={task}, mode={args.plot_mode})")
                plt.grid(True, linewidth=0.5, alpha=0.5)

                # -------- NEW: lock y-axis to be the same across tasks --------
                plt.ylim(ylim_low, ylim_high)
                # -------------------------------------------------------------

                # Safe-ish filename
                metric_safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in metric)
                save_or_show(f"metrics_vs_fraction_task_{task}_{metric_safe}_{fig_suffix()}.png")

    print("Done.")



if __name__ == "__main__":
    main()

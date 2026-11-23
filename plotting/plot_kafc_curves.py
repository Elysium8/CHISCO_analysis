"""Aggregate K-AFC results over runs and plot one curve per top-k, for each task.

Reads every k_afc_results.json under --exp-root (written by evaluation/k_afc.py).
For each task, top-k and K, accuracy is averaged over runs (seeds) and shown
with a 95% CI across runs. Relative accuracy (accuracy / chance) is computed
per run before averaging.

Writes to --out-dir (default <exp-root>/kafc_plots):
    k_afc_results_<task>_aggregate.json
    kafc_<task>_all_accuracy_curves.png
    kafc_<task>_all_relative_curves.png

    python -m plotting.plot_kafc_curves --exp-root experiments/eeg2text
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from eeg2text.eval_utils import mean_ci95, read_json, write_json


def aggregate(results: list) -> dict:
    """Mean ± 95% CI across runs of each run's (seed-averaged) accuracy."""
    acc = defaultdict(lambda: defaultdict(list))
    rel = defaultdict(lambda: defaultdict(list))
    for r in results:
        for top_k, block in r["per_topk"].items():
            for K, stats in block["per_K"].items():
                acc[top_k][K].append(stats["mean_acc"])
                rel[top_k][K].append(stats["mean_acc"] / stats["chance"])

    per_topk = {}
    for top_k in acc:
        per_K = {}
        for K in acc[top_k]:
            acc_mean, acc_ci = mean_ci95(acc[top_k][K])
            rel_mean, rel_ci = mean_ci95(rel[top_k][K])
            per_K[K] = {
                "n_runs": len(acc[top_k][K]),
                "experiment_mean_acc": acc_mean,
                "experiment_acc_ci95_pm": acc_ci,
                "experiment_mean_relative": rel_mean,
                "experiment_relative_ci95_pm": rel_ci,
            }
        per_topk[top_k] = {"per_K": per_K}
    return per_topk


def plot(per_topk: dict, value_key: str, ci_key: str, ylabel: str, path: str, chance_line: bool) -> None:
    plt.figure(figsize=(7.5, 5.2))
    for top_k in sorted(per_topk, key=int):
        per_K = per_topk[top_k]["per_K"]
        Ks = sorted(per_K, key=int)
        x = np.array([int(K) for K in Ks])
        mean = np.array([per_K[K][value_key] for K in Ks])
        ci = np.array([per_K[K][ci_key] for K in Ks])
        plt.semilogx(x, mean, marker="o", label=f"Top-{top_k}")
        plt.fill_between(x, mean - ci, mean + ci, alpha=0.18)
    if chance_line:
        plt.axhline(1.0, linestyle="--", color="gray", label="Chance")
    plt.xlabel("Candidate set size (K)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.exp_root, "kafc_plots")

    by_task = defaultdict(list)
    for dirpath, _, files in os.walk(args.exp_root):
        if "k_afc_results.json" in files:
            r = read_json(os.path.join(dirpath, "k_afc_results.json"))
            by_task[r["task"]].append(r)
    if not by_task:
        raise SystemExit(f"No k_afc_results.json found under {args.exp_root}")

    for task, results in sorted(by_task.items()):
        per_topk = aggregate(results)
        write_json(os.path.join(out_dir, f"k_afc_results_{task}_aggregate.json"),
                   {"task": task, "n_runs": len(results), "runs": [r["exp_dir"] for r in results],
                    "per_topk": per_topk})
        plot(per_topk, "experiment_mean_acc", "experiment_acc_ci95_pm", "Accuracy (%)",
             os.path.join(out_dir, f"kafc_{task}_all_accuracy_curves.png"), chance_line=False)
        plot(per_topk, "experiment_mean_relative", "experiment_relative_ci95_pm", "Relative accuracy (× chance)",
             os.path.join(out_dir, f"kafc_{task}_all_relative_curves.png"), chance_line=True)


if __name__ == "__main__":
    main()

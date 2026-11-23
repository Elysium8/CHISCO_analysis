"""Plot global-retrieval accuracy against the fraction of training data used.

Expects run folders (anywhere under --root) trained with --train-data-fraction
and evaluated with evaluation/global_retrieval.py. Accuracy is averaged over
runs (seeds) with the same task and fraction. Top-1 is left out because it is
too close to zero to read on the same axes.

    python -m plotting.plot_data_scaling --root experiments/data_scaling
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

METRICS = ("top5", "top10", "top50")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--plot-mode", choices=["absolute", "relative"], default="relative",
                    help="Plot accuracy (%%) or accuracy divided by chance.")
    ap.add_argument("--n-candidates", type=int, default=6567,
                    help="Number of candidate sentences, used for chance = k / n_candidates.")
    args = ap.parse_args()

    # values[task][fraction][metric] -> list over runs
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for dirpath, _, files in os.walk(args.root):
        if not {"config.json", "global_retrieval.json"}.issubset(files):
            continue
        with open(os.path.join(dirpath, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(dirpath, "global_retrieval.json"), encoding="utf-8") as f:
            res = json.load(f)
        for m in METRICS:
            values[cfg["task"]][float(cfg["train_data_fraction"])][m].append(res[m])

    if not values:
        raise SystemExit(f"No evaluated runs found under {args.root}")

    def transform(metric, accuracy):
        if args.plot_mode == "absolute":
            return accuracy
        k = int(metric[len("top"):])
        return accuracy / (100.0 * k / args.n_candidates)

    curves = {
        task: {m: (sorted(by_frac), [transform(m, np.mean(by_frac[f][m])) for f in sorted(by_frac)])
               for m in METRICS}
        for task, by_frac in values.items()
    }

    # One shared y-range so the read and imagine plots are directly comparable.
    all_y = [y for task in curves.values() for _, ys in task.values() for y in ys]
    pad = 0.05 * (max(all_y) - min(all_y)) or 1.0
    ylim = (min(all_y) - pad, max(all_y) + pad)

    for task, task_curves in curves.items():
        plt.figure()
        for metric, (xs, ys) in task_curves.items():
            plt.plot(xs, ys, marker="o", label=metric)
        plt.xlabel("Fraction of training data used", fontsize=14)
        plt.ylabel("Accuracy / chance" if args.plot_mode == "relative" else "Accuracy (%)", fontsize=14)
        plt.ylim(*ylim)
        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.root, f"metrics_vs_fraction_task_{task}_{args.plot_mode}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved", out)


if __name__ == "__main__":
    main()

# plot_kafc_four_panels.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_task_curves(agg, out_dir, task_label):
    task = agg["task"]
    per_topk = agg["per_topk"]

    # ---------- accuracy plot ----------
    plt.figure(figsize=(7.5, 5.2))

    for topk_str, block in sorted(per_topk.items(), key=lambda x: int(x[0])):
        topk = int(topk_str)
        perK = block["per_K"]

        Ks = np.array(sorted(int(k) for k in perK.keys()))
        means = np.array([perK[str(k)]["experiment_mean_acc"] for k in Ks])
        cis = np.array([perK[str(k)]["experiment_acc_ci95_pm"] for k in Ks])

        plt.semilogx(Ks, means, marker="o", label=f"Top-{topk}")
        plt.fill_between(Ks, means - cis, means + cis, alpha=0.18)


    plt.xlabel("Candidate set size (K)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"kafc_{task}_all_accuracy_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    # ---------- relative accuracy plot ----------
    plt.figure(figsize=(7.5, 5.2))

    for topk_str, block in sorted(per_topk.items(), key=lambda x: int(x[0])):
        topk = int(topk_str)
        perK = block["per_K"]

        Ks = np.array(sorted(int(k) for k in perK.keys()))
        means = np.array([perK[str(k)]["experiment_mean_relative"] for k in Ks])
        cis = np.array([perK[str(k)]["experiment_relative_ci95_pm"] for k in Ks])

        plt.semilogx(Ks, means, marker="o", label=f"Top-{topk}")
        plt.fill_between(Ks, means - cis, means + cis, alpha=0.18)

    plt.axhline(1.0, linestyle="--", label="Chance")

    plt.xlabel("Candidate set size (K)", fontsize=14)
    plt.ylabel("Relative accuracy (× chance)", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"kafc_{task}_all_relative_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kafc-dir",
        type=str,
        required=True,
        help="Folder containing k_afc_results_read_aggregate.json and k_afc_results_imagine_aggregate.json",
    )
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.kafc_dir, "four_panel_plots")
    os.makedirs(out_dir, exist_ok=True)

    read_path = os.path.join(args.kafc_dir, "k_afc_results_read_aggregate.json")
    recall_path = os.path.join(args.kafc_dir, "k_afc_results_imagine_aggregate.json")

    read = load_json(read_path)
    recall = load_json(recall_path)

    plot_task_curves(read, out_dir, "Read")
    plot_task_curves(recall, out_dir, "Recall")


if __name__ == "__main__":
    main()
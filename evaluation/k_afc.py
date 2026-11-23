"""K-alternative forced choice (K-AFC) evaluation.

For each test trial, the true sentence is mixed with K-1 random distractors and
the model ranks the K candidates. We report top-k accuracy as a function of K,
averaged over several random draws of distractors.

Writes to the run folder:
    k_afc_results.json
    k_afc_curve_top{k}.png            accuracy vs K
    k_afc_curve_top{k}_relative.png   accuracy / chance vs K

    python -m evaluation.k_afc --exp-dir experiments/eeg2text/read/<split>/<run>
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from eeg2text.eval_utils import Run, encode_eeg, get_device, load_text_embeddings, mean_ci95, write_json


def sample_candidates(rng: np.random.Generator, true_id: int, vocab_size: int, K: int) -> List[int]:
    """The true sentence plus K-1 distinct random distractors, in random order."""
    distractors = set()
    while len(distractors) < K - 1:
        r = int(rng.integers(0, vocab_size))
        if r != true_id:
            distractors.add(r)
    candidates = [true_id] + list(distractors)
    rng.shuffle(candidates)
    return candidates


@torch.no_grad()
def k_afc_correct(eeg_emb, true_ids, text_matrix, K: int, topks: List[int], seed: int) -> dict:
    """Number of trials whose true sentence is in the top-k of its K candidates, for each k."""
    rng = np.random.default_rng(seed)
    vocab_size = text_matrix.shape[0]
    correct = {k: 0 for k in topks}

    for emb, true_id in zip(eeg_emb, true_ids):
        if true_id is None:
            continue
        candidates = sample_candidates(rng, true_id, vocab_size, K)
        sims = (emb.unsqueeze(0) * text_matrix[candidates]).sum(dim=1)
        ranked = sims.argsort(descending=True).tolist()
        position = ranked.index(candidates.index(true_id))  # 0 = ranked first
        for k in topks:
            correct[k] += int(position < k)
    return correct


def plot_curves(Ks, means, cis, chances, top_k, out_dir):
    Ks, means, cis, chances = map(np.asarray, (Ks, means, cis, chances))

    plt.figure(figsize=(7.2, 5.0))
    plt.semilogx(Ks, means, marker="o", label="Model")
    plt.fill_between(Ks, means - cis, means + cis, alpha=0.2, label="95% CI (over seeds)")
    plt.semilogx(Ks, chances, linestyle="--", label="Chance")
    plt.xlabel("Candidate set size (K)")
    plt.ylabel(f"Top-{top_k} accuracy (%)")
    plt.title(f"K-AFC top-{top_k} accuracy vs candidate set size")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"k_afc_curve_top{top_k}.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    plt.semilogx(Ks, means / chances, marker="o", label="Model / chance")
    plt.fill_between(Ks, (means - cis) / chances, (means + cis) / chances, alpha=0.2, label="95% CI (over seeds)")
    plt.axhline(1.0, linestyle="--", label="Chance")
    plt.xlabel("Candidate set size (K)")
    plt.ylabel(f"Top-{top_k} accuracy (× chance)")
    plt.title(f"K-AFC top-{top_k} relative accuracy vs candidate set size")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"k_afc_curve_top{top_k}_relative.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="Checkpoint to use instead of <exp-dir>/best_checkpoint.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--Ks", type=int, nargs="+", default=None,
                    help="Candidate set sizes. Default: 100 500 1000 2500 and the full vocabulary.")
    ap.add_argument("--topks", type=int, nargs="+", default=[5, 10, 50])
    ap.add_argument("--n-seeds", type=int, default=1, help="Number of random distractor draws.")
    ap.add_argument("--seed-start", type=int, default=123)
    args = ap.parse_args()

    device = get_device(args.device)
    run = Run(args.exp_dir, device, checkpoint_path=args.ckpt)
    text_matrix, _, sentence_to_id = load_text_embeddings(run.sent_emb_path)
    vocab_size = text_matrix.shape[0]
    model = run.load_model(text_dim=text_matrix.shape[1])
    dataset = run.test_dataset()

    Ks = args.Ks or [100, 500, 1000, 2500, vocab_size]
    Ks = sorted({K for K in Ks if 2 <= K <= vocab_size})
    topks = sorted({k for k in args.topks if k >= 1})
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"Test trials: {len(dataset)} | vocab: {vocab_size} | Ks: {Ks} | top-k: {topks} | seeds: {seeds}")

    eeg_emb, sentences = encode_eeg(model, dataset, device, batch_size=args.eval_batch_size)
    true_ids = [sentence_to_id.get(s) for s in sentences]
    n_scored = sum(t is not None for t in true_ids)
    text_matrix = text_matrix.to(device)

    # correct[K][seed][top_k]; the same distractor draws are reused for every top-k.
    correct = {K: {s: k_afc_correct(eeg_emb, true_ids, text_matrix, K, topks, s) for s in seeds} for K in Ks}

    results = {
        "exp_dir": args.exp_dir,
        "task": run.task,
        "split_mode": run.split.get("split_mode"),
        "test_trials": len(dataset),
        "vocab_size": vocab_size,
        "Ks": Ks,
        "eval_seeds": seeds,
        "topks": topks,
        "per_topk": {},
    }

    for top_k in topks:
        per_K = {}
        for K in Ks:
            accs = [correct[K][s][top_k] / n_scored * 100.0 for s in seeds]
            mean, ci = mean_ci95(accs)
            chance = 100.0 * min(top_k, K) / K
            per_K[str(K)] = {
                "acc_per_seed": accs,
                "mean_acc": mean,
                "ci95_pm": ci,
                "chance": chance,
                "mean_correct": float(np.mean([correct[K][s][top_k] for s in seeds])),
                "total_scored": float(n_scored),
            }
            print(f"top-{top_k:<2} | K={K:5d} | acc={mean:6.2f} ± {ci:5.2f}% | chance={chance:6.2f}%")
        results["per_topk"][str(top_k)] = {"per_K": per_K}

        plot_curves(Ks, [per_K[str(K)]["mean_acc"] for K in Ks], [per_K[str(K)]["ci95_pm"] for K in Ks],
                    [per_K[str(K)]["chance"] for K in Ks], top_k, args.exp_dir)

    write_json(os.path.join(args.exp_dir, "k_afc_results.json"), results)
    print("Saved k_afc_results.json and plots to", args.exp_dir)


if __name__ == "__main__":
    main()

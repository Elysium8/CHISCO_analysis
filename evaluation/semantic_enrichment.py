"""Semantic enrichment of retrieved sentences.

Every CHISCO sentence belongs to a semantic category. For each test trial we
count how many of its top-k retrieved sentences (excluding the target itself)
share the target's category, and compare that count with the hypergeometric
expectation under random retrieval. Counts are pooled over trials and tested
with a one-sided normal approximation.

Single run   -> writes global_retrieval_semantic_test.json to the run folder.
Many runs    -> also writes <exp-root>/semantic_aggregate/semantic_aggregate_summary.json
                with mean ± 95% CI over runs, grouped by task.

    python -m evaluation.semantic_enrichment --exp-dir <run folder>
    python -m evaluation.semantic_enrichment --exp-root experiments/eeg2text
"""

import argparse
import math
import os
from collections import defaultdict
from typing import Dict, List

from eeg2text.eval_utils import (
    Run, encode_eeg, find_run_dirs, get_device, load_text_embeddings, mean_ci95, read_json, top_k_retrieval,
    write_json,
)


def load_sentence_classes(textmaps_path: str, sentences: List[str]) -> Dict[str, int]:
    """Map each candidate sentence to its integer category from textmaps.json."""
    textmaps = read_json(textmaps_path)
    out = {}
    for s in sentences:
        try:
            out[s] = int(textmaps[s])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def semantic_enrichment(candidates, true_ids, sentences, id_to_sentence, sentence_class, topn, topk, include_target):
    """Same-category counts in the top-n / top-k candidates vs the hypergeometric null."""
    class_counts = defaultdict(int)
    for c in sentence_class.values():
        class_counts[c] += 1
    N = len(sentence_class)

    hits_n = total_n = hits_k = total_k = 0
    null_mean = null_var = 0.0

    for target, true_id, cand in zip(sentences, true_ids, candidates):
        target_class = sentence_class.get(target)
        if true_id is None or target_class is None:
            continue

        def same_class(sentence):
            return (include_target or sentence != target) and sentence_class.get(sentence) == target_class

        top_n = [id_to_sentence[j] for j in cand[:topn]]
        top_k = [id_to_sentence[j] for j in cand[:topk]]
        hits_n += sum(map(same_class, top_n))
        total_n += len(top_n)
        hits_k += sum(map(same_class, top_k))
        total_k += len(top_k)

        # Hypergeometric mean and variance of same-category draws in len(top_k) picks.
        p = class_counts[target_class] / N
        n = len(top_k)
        null_mean += n * p
        null_var += n * p * (1 - p) * (N - n) / (N - 1)

    z = (hits_k - null_mean) / math.sqrt(null_var) if null_var > 0 else float("nan")
    return {
        "semantic_topn": topn,
        "semantic_topk_test": topk,
        "include_target_in_semantic": include_target,
        "topn_same_class_fraction": hits_n / total_n if total_n else float("nan"),
        "topk_same_class_fraction": hits_k / total_k if total_k else float("nan"),
        "topk_same_class_count": hits_k,
        "topk_total_count": total_k,
        "topk_expected_same_class_count": null_mean,
        "topk_enrichment_vs_chance": hits_k / null_mean if null_mean > 0 else float("inf"),
        "z": z,
        "p_value_one_sided": 0.5 * math.erfc(z / math.sqrt(2)),
    }


def evaluate_run(run_dir: str, args, device) -> dict:
    run = Run(run_dir, device)
    text_matrix, id_to_sentence, sentence_to_id = load_text_embeddings(run.sent_emb_path)
    sentence_class = load_sentence_classes(args.textmaps, id_to_sentence)
    model = run.load_model(text_dim=text_matrix.shape[1])
    dataset = run.test_dataset()

    print(f"[{os.path.basename(run_dir)}] task={run.task} | test trials={len(dataset)} | "
          f"sentences with a category: {len(sentence_class)}/{len(id_to_sentence)}")

    eeg_emb, sentences = encode_eeg(model, dataset, device, batch_size=args.eval_batch_size)
    ks = sorted({1, 5, 10, 50, args.semantic_topn, args.semantic_topk})
    accuracy, candidates, true_ids = top_k_retrieval(eeg_emb, sentences, text_matrix, sentence_to_id, ks)

    result = {
        **{f"top{k}": accuracy[f"top{k}"] for k in (1, 5, 10, 50)},
        "total": float(len(sentences)),
        "semantic": semantic_enrichment(candidates, true_ids, sentences, id_to_sentence, sentence_class,
                                        args.semantic_topn, args.semantic_topk, args.include_target_in_semantic),
        "exp_dir": run_dir,
        "task": run.task,
        "split_mode": run.split.get("split_mode"),
        "eval_split": "test",
    }
    write_json(os.path.join(run_dir, "global_retrieval_semantic_test.json"), result)
    return result


def summarize(results: List[dict]) -> dict:
    metrics = {
        "top1": [r["top1"] for r in results],
        "top5": [r["top5"] for r in results],
        "top10": [r["top10"] for r in results],
        "top50": [r["top50"] for r in results],
        "same_class_topn_fraction_percent": [100 * r["semantic"]["topn_same_class_fraction"] for r in results],
        "same_class_topk_fraction_percent": [100 * r["semantic"]["topk_same_class_fraction"] for r in results],
        "enrichment_vs_chance": [r["semantic"]["topk_enrichment_vs_chance"] for r in results],
        "z": [r["semantic"]["z"] for r in results],
        "p_value_one_sided": [r["semantic"]["p_value_one_sided"] for r in results],
    }
    summary = {"n_runs": len(results), "run_dirs": [r["exp_dir"] for r in results], "metrics": {}}
    for name, values in metrics.items():
        mean, ci = mean_ci95(values)
        summary["metrics"][name] = {"mean": mean, "ci95_pm": ci, "values": values}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--exp-dir", help="A single run folder")
    where.add_argument("--exp-root", help="Folder searched recursively for run folders")
    ap.add_argument("--out-dir", default=None, help="Default: <exp-root>/semantic_aggregate")
    ap.add_argument("--textmaps", default="./ds005170/json/textmaps.json", help="sentence -> category id")
    ap.add_argument("--semantic-topn", type=int, default=10)
    ap.add_argument("--semantic-topk", type=int, default=50, help="Top-k used for the enrichment test")
    ap.add_argument("--include-target-in-semantic", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
    args = ap.parse_args()

    device = get_device(args.device)
    run_dirs = [args.exp_dir] if args.exp_dir else find_run_dirs(args.exp_root)
    print(f"Evaluating {len(run_dirs)} run(s)")

    by_task = defaultdict(list)
    for run_dir in run_dirs:
        r = evaluate_run(run_dir, args, device)
        by_task[r["task"]].append(r)
        sem = r["semantic"]
        print(f"  top-1 {r['top1']:.2f}% | top-10 {r['top10']:.2f}% | "
              f"same-class top-{sem['semantic_topk_test']} {100 * sem['topk_same_class_fraction']:.2f}% | "
              f"enrichment {sem['topk_enrichment_vs_chance']:.2f}x | p = {sem['p_value_one_sided']:.3g}")

    if args.exp_root:
        summary = {
            "exp_root": args.exp_root,
            "semantic_topn": args.semantic_topn,
            "semantic_topk": args.semantic_topk,
            "groups": {task: summarize(results) for task, results in sorted(by_task.items())},
        }
        out_path = os.path.join(args.out_dir or os.path.join(args.exp_root, "semantic_aggregate"),
                                "semantic_aggregate_summary.json")
        write_json(out_path, summary)
        print("\nSaved", out_path)

        for task, block in summary["groups"].items():
            print(f"\n=== {task} ({block['n_runs']} runs) ===")
            for name, stats in block["metrics"].items():
                print(f"{name}: {stats['mean']:.3f} ± {stats['ci95_pm']:.3f}")


if __name__ == "__main__":
    main()

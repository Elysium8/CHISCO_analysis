"""Empirical chance level: global retrieval when the EEG is replaced by Gaussian noise.

Each test trial keeps its subject ID and target sentence, but its EEG is
replaced with standard-normal noise of the same shape, then scored with the
trained model exactly like evaluation/global_retrieval.py.

Single run -> writes global_retrieval_random_noise.json to the run folder.
Many runs  -> also writes <exp-root>/random_noise_summary.json (mean ± 95% CI per task).

    python -m evaluation.random_noise_baseline --exp-root experiments/eeg2text
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg2text.eval_utils import (
    Run, encode_eeg, find_run_dirs, get_device, load_text_embeddings, mean_ci95, top_k_retrieval, write_json,
)

KS = (1, 5, 10, 50)


class RandomNoiseDataset(Dataset):
    def __init__(self, refs, n_channels: int, n_samples: int, seed: int = 0):
        self.refs = refs
        self.shape = (n_channels, n_samples)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        ref = self.refs[idx]
        return {
            "eeg": torch.from_numpy(self.rng.standard_normal(self.shape, dtype=np.float32)),
            "subject_id": torch.tensor(ref.subject_id, dtype=torch.long),
            "sentence": ref.sentence,
        }


def evaluate_run(run_dir: str, args, device) -> dict:
    run = Run(run_dir, device)
    text_matrix, _, sentence_to_id = load_text_embeddings(run.sent_emb_path)
    model = run.load_model(text_dim=text_matrix.shape[1])

    test = run.test_dataset()
    n_channels, n_samples = test[0]["eeg"].shape
    noise = RandomNoiseDataset(test.refs, n_channels, n_samples, seed=args.noise_seed)

    eeg_emb, sentences = encode_eeg(model, noise, device, batch_size=args.batch_size)
    accuracy, _, _ = top_k_retrieval(eeg_emb, sentences, text_matrix, sentence_to_id, KS)

    result = {
        "exp_dir": run_dir,
        "task": run.task,
        "eval_split": "test",
        "noise_seed": args.noise_seed,
        "eeg_shape": [n_channels, n_samples],
        **accuracy,
        "total": float(len(sentences)),
    }
    write_json(os.path.join(run_dir, "global_retrieval_random_noise.json"), result)
    print(f"[{os.path.basename(run_dir)}] task={run.task} | "
          + " | ".join(f"top-{k} {accuracy[f'top{k}']:.3f}%" for k in KS))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--exp-dir", help="A single run folder")
    where.add_argument("--exp-root", help="Folder searched recursively for run folders")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--out-summary", default=None, help="Default: <exp-root>/random_noise_summary.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = get_device(args.device)
    run_dirs = [args.exp_dir] if args.exp_dir else find_run_dirs(args.exp_root)
    print(f"Evaluating {len(run_dirs)} run(s)")

    by_task = defaultdict(list)
    for run_dir in run_dirs:
        r = evaluate_run(run_dir, args, device)
        by_task[r["task"]].append(r)

    if args.exp_root:
        summary = {"exp_root": args.exp_root, "noise_seed": args.noise_seed, "groups": {}}
        for task, results in sorted(by_task.items()):
            print(f"\n=== {task} ({len(results)} runs) ===")
            metrics = {}
            for k in KS:
                values = [r[f"top{k}"] for r in results]
                mean, ci = mean_ci95(values)
                metrics[f"top{k}"] = {"mean": mean, "ci95_pm": ci, "values": values}
                print(f"top{k}: {mean:.3f} ± {ci:.3f}")
            summary["groups"][task] = {"n_runs": len(results), "run_dirs": [r["exp_dir"] for r in results],
                                       "metrics": metrics}

        out_path = args.out_summary or os.path.join(args.exp_root, "random_noise_summary.json")
        write_json(out_path, summary)
        print("\nSaved", out_path)


if __name__ == "__main__":
    main()

"""Global retrieval: rank every sentence in the corpus for each test trial.

Writes to the run folder:
    global_retrieval.json         top-1/5/10/50 accuracy (%)
    global_retrieval_trials.json  per-trial correctness, for paired comparisons between models

    python -m evaluation.global_retrieval --exp-dir experiments/eeg2text/read/<split>/<run>
"""

import argparse
import os

from eeg2text.eval_utils import Run, encode_eeg, get_device, load_text_embeddings, top_k_retrieval, write_json

KS = (1, 5, 10, 50)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True, help="Run folder containing config.json, split.json and best_checkpoint.pt")
    ap.add_argument("--ckpt", default=None, help="Checkpoint to use instead of <exp-dir>/best_checkpoint.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
    args = ap.parse_args()

    device = get_device(args.device)
    run = Run(args.exp_dir, device, checkpoint_path=args.ckpt)
    text_matrix, _, sentence_to_id = load_text_embeddings(run.sent_emb_path)
    model = run.load_model(text_dim=text_matrix.shape[1])
    dataset = run.test_dataset()

    print(f"Task: {run.task} | test trials: {len(dataset)} | candidate sentences: {len(sentence_to_id)}")

    eeg_emb, sentences = encode_eeg(model, dataset, device, batch_size=args.eval_batch_size)
    accuracy, candidates, true_ids = top_k_retrieval(eeg_emb, sentences, text_matrix, sentence_to_id, KS)

    trial_rows = []
    for n, (ref, true_id, cand) in enumerate(zip(dataset.refs, true_ids, candidates)):
        found = true_id is not None and true_id in cand
        trial_rows.append({
            "trial_number": n,
            "trial_id": ref.trial_id,
            "sub_code": ref.sub_code,
            "run_idx": ref.run_idx,
            "trial_in_run": ref.trial_in_run,
            "sentence": ref.sentence,
            "true_id": true_id,
            "rank_within_top50": cand.index(true_id) + 1 if found else None,
            **{f"top{k}_correct": int(found and true_id in cand[:k]) for k in KS},
        })

    summary = {**accuracy, "total": float(len(sentences))}
    write_json(os.path.join(args.exp_dir, "global_retrieval.json"), summary)
    write_json(os.path.join(args.exp_dir, "global_retrieval_trials.json"), trial_rows)

    for k in KS:
        print(f"Top-{k:<3}: {accuracy[f'top{k}']:.2f}%")
    print(f"(over {len(sentences)} test trials)")
    print("Saved global_retrieval.json and global_retrieval_trials.json to", args.exp_dir)


if __name__ == "__main__":
    main()

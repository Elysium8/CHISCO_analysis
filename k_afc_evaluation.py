# k_afc_evaluation.py
#
# K-way forced-choice (K-AFC) evaluation:
# For each EEG segment in the validation set, build a candidate set of size K:
#   - 1 correct sentence
#   - K-1 randomly sampled incorrect sentences (from full vocab)
# Predict the candidate with highest cosine similarity to EEG embedding.
#
# MODIFICATION (minimal):
#   - Added --top-k argument (default=1) to compute Top-k accuracy instead of Top-1.
#
# This script is based ENTIRELY on the same working logic as your binary
# discrimination script (choose_one_testing.py):
#   - streaming index build
#   - split reconstruction from split.json
#   - sentence embedding loading from JSON (utf-8)
#   - model construction from checkpoint/config args
#
# Outputs:
#   - k_afc_results.json (per K, per seed, mean, CI)
#   - k_afc_curve.png (accuracy vs K with chance line)
#
# Usage:
#   python k_afc_evaluation.py --exp-dir <run_dir>
#   python k_afc_evaluation.py --exp-dir <run_dir> --n-seeds 3
#   python k_afc_evaluation.py --exp-dir <run_dir> --Ks 2 5 10 25 50 100 500 1000
#   python k_afc_evaluation.py --exp-dir <run_dir> --top-k 10
#
import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib.pyplot as plt

# Same import style as your working scripts
sys.path.append("./ds005170")
from data_loader import load_read_run, load_imagine_run  # type: ignore

# Same model import pattern
try:
    from updated_train_eeg_to_text_embed import EEGToTextEmbedModel
except Exception:
    from train_eeg_to_text_embed import EEGToTextEmbedModel


# -------------------- helpers (same as working script) --------------------

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sentence_embeddings(path: str) -> Dict[str, List[float]]:
    # EXACTLY the working approach: JSON + utf-8
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_text_embedding_matrix(
    sent2emb: Dict[str, List[float]]
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    sentences = sorted(sent2emb.keys())
    sentence_to_id = {s: i for i, s in enumerate(sentences)}
    matrix = np.stack([np.array(sent2emb[s], dtype=np.float32) for s in sentences], axis=0)
    matrix = torch.tensor(matrix)  # (N_sent, D)
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    return matrix, sentences, sentence_to_id

def make_train_val_indices(n: int, val_split: float, shuffle: bool, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    val_size = int(n * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return train_idx, val_idx

def split_indices_from_splitinfo(
    split_info: Dict[str, Any],
    sentences: List[str],
    subj_ids: torch.Tensor,
    subject_codes: List[str],
    seed: int,
    val_split: float,
    shuffle: bool,
) -> Tuple[List[int], List[int]]:
    """
    Same split reconstruction logic as your working script:
    supports iid / unseen_sentence / unseen_subject.
    """
    mode = split_info.get("split_mode", "iid")
    n = len(sentences)

    if mode == "iid":
        train_idx, val_idx = make_train_val_indices(n, val_split=val_split, shuffle=shuffle, seed=seed)
        return train_idx, val_idx

    if mode == "unseen_sentence":
        holdout_sents = split_info.get("holdout_sentences", None)
        if holdout_sents is None:
            frac = float(split_info["holdout_fraction"])
            rng = random.Random(seed)
            unique = sorted(set(sentences))
            rng.shuffle(unique)
            n_holdout = max(1, int(len(unique) * frac))
            holdout_sents = set(unique[:n_holdout])
        else:
            holdout_sents = set(holdout_sents)

        val_idx = [i for i, s in enumerate(sentences) if s in holdout_sents]
        train_idx = [i for i, s in enumerate(sentences) if s not in holdout_sents]
        return train_idx, val_idx

    if mode == "unseen_subject":
        holdout_codes = split_info.get("holdout_subjects", None)
        if holdout_codes is None:
            raise ValueError("split.json missing holdout_subjects for unseen_subject evaluation.")

        code_to_idx = {c: i for i, c in enumerate(subject_codes)}
        holdout_ids = {code_to_idx[c] for c in holdout_codes}

        val_idx = [i for i in range(n) if int(subj_ids[i]) in holdout_ids]
        train_idx = [i for i in range(n) if int(subj_ids[i]) not in holdout_ids]
        return train_idx, val_idx

    raise ValueError(f"Unknown split_mode in split.json: {mode}")


# -------------------- streaming dataset (same as working script) --------------------

@dataclass(frozen=True)
class TrialRef:
    sub_code: str
    run_idx: int
    trial_in_run: int
    subject_index: int
    sentence: str
    task: str  # "read" or "imagine"

class ChiscoStreamingEvalDataset(Dataset):
    """
    Streaming dataset: stores only refs; loads run on-demand in __getitem__.
    """
    def __init__(self, root: str, trial_index: List[TrialRef]):
        self.root = root
        self.trial_index = trial_index
        self._cache_key: Optional[Tuple[str, int, str]] = None
        self._cache_trials: Optional[List[dict]] = None

    def __len__(self) -> int:
        return len(self.trial_index)

    def _load_run_trials(self, sub_code: str, run_idx: int, task: str) -> List[dict]:
        key = (sub_code, run_idx, task)
        if self._cache_key == key and self._cache_trials is not None:
            return self._cache_trials

        if task == "read":
            trials = load_read_run(self.root, sub_code, run_idx)
        elif task == "imagine":
            trials = load_imagine_run(self.root, sub_code, run_idx)
        else:
            raise ValueError(f"Unknown task: {task}")

        self._cache_key = key
        self._cache_trials = trials
        return trials

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self.trial_index[idx]
        trials = self._load_run_trials(ref.sub_code, ref.run_idx, ref.task)
        trial = trials[ref.trial_in_run]

        arr = trial["input_features"]  # (1, 125, 1651), float64
        eeg = arr[0, :122, :] * 1_000_000.0
        eeg = eeg.astype(np.float32)
        eeg = torch.from_numpy(eeg)  # (122, 1651)

        return {
            "eeg": eeg,
            "subject_id": torch.tensor(ref.subject_index, dtype=torch.long),
            "sentence": ref.sentence,
        }

def build_streaming_index(
    root: str,
    subject_codes: List[str],
    task: str,
    n_runs_read: int = 45,
    n_runs_imagine: int = 45,
) -> Tuple[List[TrialRef], List[str], torch.Tensor]:
    trial_index: List[TrialRef] = []
    sentences: List[str] = []
    subj_ids: List[int] = []

    if task == "read":
        n_runs = n_runs_read
        run_loader = load_read_run
    elif task == "imagine":
        n_runs = n_runs_imagine
        run_loader = load_imagine_run
    else:
        raise ValueError(f"Unknown task: {task}")

    for subj_idx, sub_code in enumerate(subject_codes):
        for run_idx in range(1, n_runs + 1):
            trials = run_loader(root, sub_code, run_idx)
            if not trials:
                continue
            for t_i, trial in enumerate(trials):
                sent = str(trial["text"]).strip()
                trial_index.append(
                    TrialRef(
                        sub_code=sub_code,
                        run_idx=run_idx,
                        trial_in_run=t_i,
                        subject_index=subj_idx,
                        sentence=sent,
                        task=task,
                    )
                )
                sentences.append(sent)
                subj_ids.append(subj_idx)

    subj_ids_t = torch.tensor(subj_ids, dtype=torch.long)
    return trial_index, sentences, subj_ids_t


# -------------------- K-AFC evaluation --------------------

def _sample_k_candidates(rng: np.random.Generator, true_id: int, vocab_size: int, K: int) -> List[int]:
    """
    Candidates: [true_id] + (K-1) random distinct negatives from full vocab.
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    if vocab_size < K:
        raise ValueError(f"Vocab size ({vocab_size}) must be >= K ({K})")

    negatives: Set[int] = set()
    while len(negatives) < (K - 1):
        r = int(rng.integers(0, vocab_size))
        if r != true_id:
            negatives.add(r)

    cand = [true_id] + list(negatives)
    rng.shuffle(cand)
    return cand

@torch.no_grad()
def evaluate_k_afc(
    model,
    dataset: Dataset,
    text_matrix: torch.Tensor,
    sentence_to_id: Dict[str, int],
    device: torch.device,
    K: int,
    seed: int,
    batch_size: int,
    top_k: int = 1,  # NEW
) -> Dict[str, float]:
    """
    For each trial:
      - sample K candidates (1 true + K-1 random negatives)
      - choose top-k by cosine similarity; correct if true is in top-k
    """
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")

    model.eval()
    text_matrix = text_matrix.to(device)
    rng = np.random.default_rng(seed)
    vocab_size = int(text_matrix.shape[0])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    skipped_missing_label = 0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        batch_sentences = batch["sentence"]

        eeg_emb = model(eeg, subj)
        eeg_emb = torch.nn.functional.normalize(eeg_emb, dim=1)  # (B, D)

        for i, s in enumerate(batch_sentences):
            tid = sentence_to_id.get(s, None)
            if tid is None:
                skipped_missing_label += 1
                continue

            candidates = _sample_k_candidates(rng, int(tid), vocab_size, K)
            cand_ids = torch.tensor(candidates, device=device, dtype=torch.long)

            # cosine sim because both normalized
            sims = (eeg_emb[i:i+1] * text_matrix[cand_ids]).sum(dim=1)  # (K,)

            k_eff = min(int(top_k), len(candidates))
            topk_idx = torch.topk(sims, k=k_eff, largest=True).indices.tolist()
            topk_ids = {candidates[j] for j in topk_idx}

            correct += int(int(tid) in topk_ids)
            total += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    return {
        "acc": float(acc),
        "correct": float(correct),
        "total": float(total),
        "skipped_missing_label": float(skipped_missing_label),
        "chance": float(100.0 / K),  # NOTE: this remains top-1 chance; interpret accordingly for top-k
    }


def _mean_ci95(accs: List[float]) -> Tuple[float, float]:
    """
    mean ± 95% CI half-width (normal approx) across seeds
    """
    if len(accs) == 0:
        return 0.0, 0.0
    mean = float(np.mean(accs))
    if len(accs) == 1:
        return mean, 0.0
    std = float(np.std(accs, ddof=1))
    sem = std / float(np.sqrt(len(accs)))
    return mean, 1.96 * sem


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="K-AFC evaluation (based on working binary discrimination script).")
    ap.add_argument("--exp-dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eval-batch-size", type=int, default=32)

    ap.add_argument("--n-seeds", type=int, default=3, help="Default 3 to keep runtime reasonable.")
    ap.add_argument("--seed-start", type=int, default=123)

    ap.add_argument(
        "--Ks",
        type=int,
        nargs="+",
        default=None,
        help="Candidate sizes K. Default: 2 5 10 25 50 100 500 1000 FULLVOCAB",
    )

    # NEW: top-k accuracy control
    ap.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k accuracy (default: 1 = standard top-1).",
    )

    args = ap.parse_args()

    exp_dir = args.exp_dir
    config_path = os.path.join(exp_dir, "config.json")
    split_path = os.path.join(exp_dir, "split.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {exp_dir}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing split.json in {exp_dir}")

    cfg = read_json(config_path)
    split_info = read_json(split_path)

    ckpt_path = args.ckpt or os.path.join(exp_dir, "best_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Using device:", device, flush=True)

    chisco_root = cfg.get("chisco_root") or cfg.get("CHISCO_ROOT") or "./ds005170/derivatives/preprocessed_pkl"
    sent_emb_path = cfg.get("sent_emb_path") or cfg.get("SENT_EMB_PATH")
    if sent_emb_path is None:
        raise ValueError("config.json missing sent_emb_path")
    task = cfg.get("task") or cfg.get("TASK") or "read"

    subject_codes_str = cfg.get("subject_codes")
    if isinstance(subject_codes_str, str):
        subject_codes = [s.strip() for s in subject_codes_str.split(",") if s.strip()]
    elif isinstance(subject_codes_str, list):
        subject_codes = subject_codes_str
    else:
        subject_codes = cfg.get("SUBJECT_CODES") or [f"{i:02d}" for i in range(1, 6)]

    seed_split = int(cfg.get("seed") or cfg.get("RANDOM_SEED") or split_info.get("seed") or 42)
    val_split = float(cfg.get("val_split") or cfg.get("VAL_SPLIT") or 0.2)
    shuffle = bool(cfg.get("shuffle_data") if "shuffle_data" in cfg else cfg.get("SHUFFLE_DATA", True))

    # 1) Build streaming index (same)
    print("Building streaming index (this can take a while)...", flush=True)
    trial_index, sentences, subj_ids = build_streaming_index(
        root=chisco_root,
        subject_codes=subject_codes,
        task=task,
    )
    print(f"Index built: {len(trial_index)} trials", flush=True)

    # 2) Reconstruct split (same)
    _, val_idx = split_indices_from_splitinfo(
        split_info=split_info,
        sentences=sentences,
        subj_ids=subj_ids,
        subject_codes=subject_codes,
        seed=seed_split,
        val_split=val_split,
        shuffle=shuffle,
    )
    print(f"Val trials: {len(val_idx)}", flush=True)

    # 3) Load sentence embeddings from JSON (same)
    sent2emb = load_sentence_embeddings(sent_emb_path)
    text_matrix, id_to_sentence, sentence_to_id = build_text_embedding_matrix(sent2emb)
    vocab_size = int(text_matrix.shape[0])
    text_dim = int(text_matrix.shape[1])
    print(f"Text vocab size: {vocab_size} | text_dim={text_dim}", flush=True)

    # Choose Ks
    if args.Ks is None:
        Ks = [100, 500, 1000, 2500, vocab_size]
    else:
        Ks = list(args.Ks)
        if vocab_size not in Ks:
            # keep it optional; don't force
            pass

    Ks = [k for k in Ks if 2 <= k <= vocab_size]
    Ks = sorted(set(Ks))

    eval_seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_seeds)))
    print(f"Ks        : {Ks}", flush=True)
    print(f"Eval seeds: {eval_seeds}", flush=True)
    print(f"Top-k     : {int(args.top_k)}", flush=True)

    # 4) Load model from checkpoint/config (same explicit-args approach as working script)
    checkpoint = torch.load(ckpt_path, map_location=device)
    c2 = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    def _get(name: str, default):
        return c2.get(name, cfg.get(name, default))

    n_channels = 122
    n_subjects = len(subject_codes)

    latent_dim = int(_get("latent_dim", _get("LATENT_DIM", 256)))
    hidden_channels = int(_get("hidden_channels", _get("HIDDEN_CHANNELS", 256)))
    final_hidden = int(_get("final_hidden", _get("FINAL_HIDDEN", 256)))
    n_blocks = int(_get("n_blocks", _get("N_BLOCKS", 5)))
    pooling = _get("pooling", _get("POOLING", "mean"))

    use_eegnet = bool(_get("use_eegnet", True))
    eegnet_temporal_kern = int(_get("eegnet_temporal_kern", 64))
    eegnet_f1 = int(_get("eegnet_f1", 16))
    eegnet_d = int(_get("eegnet_d", 2))
    raw_frontend_out_ch = int(_get("raw_frontend_out_ch", 32))

    use_spatial_attn = bool(_get("use_spatial_attn", True))
    use_subject_layer = bool(_get("use_subject_layer", True))
    use_residual_blocks = bool(_get("use_residual_blocks", True))

    model = EEGToTextEmbedModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        text_dim=text_dim,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        final_hidden=final_hidden,
        n_blocks=n_blocks,
        pooling=pooling,
        use_eegnet=use_eegnet,
        eegnet_temporal_kern=eegnet_temporal_kern,
        eegnet_F1=eegnet_f1,
        eegnet_D=eegnet_d,
        use_spatial_attn=use_spatial_attn,
        use_subject_layer=use_subject_layer,
        use_residual_blocks=use_residual_blocks,
        raw_frontend_out_ch=raw_frontend_out_ch,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # 5) Build val dataset (same)
    base_ds = ChiscoStreamingEvalDataset(root=chisco_root, trial_index=trial_index)
    val_ds = Subset(base_ds, val_idx)

    # 6) Run K-AFC
    results: Dict[str, Any] = {
        "exp_dir": exp_dir,
        "task": task,
        "split_mode": split_info.get("split_mode", "iid"),
        "val_trials": len(val_idx),
        "vocab_size": vocab_size,
        "Ks": Ks,
        "eval_seeds": eval_seeds,
        "top_k": int(args.top_k),  # NEW: record in results
        "per_K": {},
    }

    print("\nRunning K-AFC...", flush=True)
    for K in Ks:
        per_seed = []
        per_seed_correct = []
        total_scored = None
        chance = 100.0 / K  # NOTE: still top-1 chance baseline

        for s in eval_seeds:
            r = evaluate_k_afc(
                model=model,
                dataset=val_ds,
                text_matrix=text_matrix,
                sentence_to_id=sentence_to_id,
                device=device,
                K=K,
                seed=s,
                batch_size=args.eval_batch_size,
                top_k=args.top_k,  # NEW
            )
            per_seed.append(r["acc"])
            per_seed_correct.append(r["correct"])
            total_scored = r["total"]  # should be constant

        mean, pm = _mean_ci95(per_seed)
        mean_correct = float(np.mean(per_seed_correct)) if len(per_seed_correct) else 0.0

        results["per_K"][str(K)] = {
            "acc_per_seed": per_seed,
            "mean_acc": mean,
            "ci95_pm": pm,
            "chance": chance,
            "mean_correct": mean_correct,
            "total_scored": total_scored,
        }

        print(
            f"K={K:5d} | top-{int(args.top_k)} acc={mean:6.2f} ± {pm:5.2f}% | "
            f"chance(top-1)={chance:6.2f}% | avg correct={mean_correct:.1f}/{int(total_scored)}",
            flush=True,
        )

    # Save JSON
    out_json = os.path.join(exp_dir, "k_afc_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    # 7) Plot
    Ks_arr = np.array([int(k) for k in results["per_K"].keys()], dtype=np.int64)
    Ks_arr.sort()

    means = np.array([results["per_K"][str(int(k))]["mean_acc"] for k in Ks_arr], dtype=np.float32)
    cis = np.array([results["per_K"][str(int(k))]["ci95_pm"] for k in Ks_arr], dtype=np.float32)
    chances = np.array([results["per_K"][str(int(k))]["chance"] for k in Ks_arr], dtype=np.float32)

    plt.figure(figsize=(7.2, 5.0))
    plt.semilogx(Ks_arr, means, marker="o", label="Model")
    plt.fill_between(Ks_arr, means - cis, means + cis, alpha=0.2, label="95% CI (over seeds)")
    plt.semilogx(Ks_arr, chances, linestyle="--", label="Chance (1/K)")

    plt.xlabel("Candidate set size (K)")
    plt.ylabel(f"Top-{int(args.top_k)} accuracy (%)")
    plt.title(f"K-AFC EEG→Text Top-{int(args.top_k)} performance vs candidate vocabulary size")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    out_png = os.path.join(exp_dir, "k_afc_curve.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}", flush=True)

    acc_over_chance = means / chances

    plt.figure(figsize=(7.2, 5.0))
    plt.semilogx(Ks_arr, acc_over_chance, marker="o")

    plt.xlabel("Candidate set size (K)")
    plt.ylabel("Accuracy ÷ Chance")
    plt.title(f"K-AFC EEG→Text Top-{int(args.top_k)} performance relative to chance (top-1 baseline)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    out_png_ratio = os.path.join(exp_dir, "k_afc_curve_over_chance.png")
    plt.tight_layout()
    plt.savefig(out_png_ratio, dpi=200)
    plt.close()
    print(f"Saved: {out_png_ratio}", flush=True)


if __name__ == "__main__":
    main()

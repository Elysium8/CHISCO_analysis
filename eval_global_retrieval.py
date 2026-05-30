# eval_global_retrieval.py
#
# Evaluate EEG→text embeddings via *global retrieval* using one saved experiment folder.
#
# Usage:
#   python eval_global_retrieval.py --exp-dir experiments/eeg2text/read/iid/<run_id>
#
# Functionality is intended to be IDENTICAL to the original version:
#   - loads best_checkpoint.pt
#   - loads config.json and split.json from that run folder
#   - reconstructs the exact validation set for:
#       iid / unseen_sentence / unseen_subject
#   - evaluates global retrieval against ALL sentence embeddings in SENT_EMB_PATH
#
# The ONLY change is memory behavior:
#   - Instead of loading X=(N,C,T) into RAM, we build a streaming index and load trials on-demand.

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Make ds005170 importable
sys.path.append("./ds005170")

# Reuse the same low-level loaders (so preprocessing stays consistent)
from data_loader import load_read_run, load_imagine_run  # type: ignore

from train_script_with_split import EEGToTextEmbedModel, _CHISCOBaselineWrapper
from CHISCO_baseline_model import CHISCOBaselineEEGEncoder

# -------------------- helpers --------------------

def normalize_sentence(s):
    return str(s).strip()

def indices_from_sentence_split(sentences, split_sentences):
    split_set = set(normalize_sentence(s) for s in split_sentences)
    return [i for i, s in enumerate(sentences) if normalize_sentence(s) in split_set]

def split_indices_from_splitinfo(split_info, sentences):
    if "train_sentences" in split_info:
        train_idx = indices_from_sentence_split(sentences, split_info["train_sentences"])
        val_idx = indices_from_sentence_split(sentences, split_info["val_sentences"])
        test_idx = indices_from_sentence_split(sentences, split_info["test_sentences"])
        return train_idx, val_idx, test_idx

    if "train_idx" in split_info and "val_idx" in split_info:
        return split_info["train_idx"], split_info["val_idx"], split_info.get("test_idx", [])

    raise KeyError("split.json must contain sentence splits or index splits.")

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sentence_embeddings(path: str) -> Dict[str, List[float]]:
    # Keep identical behavior: read the json map sentence -> embedding(list[float])
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


# def split_indices_from_splitinfo(
#         split_info: Dict[str, Any],
#         sentences: List[str],
#         subj_ids: torch.Tensor,
#         subject_codes: List[str],
#         seed: int,
#         val_split: float,
#         shuffle: bool,
# ) -> Tuple[List[int], List[int]]:
#     """
#     Force the use of explicit indices saved in split.json.
#     Throws an error if they are missing to prevent split mismatch.
#     """
#     # Force usage of saved indices
#     if "train_idx" in split_info and "val_idx" in split_info:
#         train_idx = split_info["train_idx"]
#         val_idx = split_info["val_idx"]
#
#         # Basic sanity check
#         if not val_idx:
#             raise ValueError("The 'val_idx' list in split.json is empty.")
#
#         return train_idx, val_idx
#
#     # If the keys are missing, we throw a hard error instead of guessing the RNG
#     mode = split_info.get("split_mode", "unknown")
#     raise KeyError(
#         f"CRITICAL ERROR: 'train_idx' or 'val_idx' not found in split.json. "
#         f"Evaluation for mode '{mode}' cannot proceed safely because the "
#         f"original training indices are required to prevent data leakage or set mismatch."
#     )


# def split_indices_from_splitinfo(
#     split_info: Dict[str, Any],
#     sentences: List[str],
#     subj_ids: torch.Tensor,
#     subject_codes: List[str],
#     seed: int,
#     val_split: float,
#     shuffle: bool,
# ) -> Tuple[List[int], List[int]]:
#     """
#     Recreate the split based on split.json.
#
#     (Kept identical to your original logic.)
#     """
#     mode = split_info.get("split_mode", "iid")
#     n = len(sentences)
#
#     if mode == "iid":
#         _, val_idx = make_train_val_indices(n, val_split=val_split, shuffle=shuffle, seed=seed)
#         return list(range(n))[int(n * val_split):], val_idx  # train_idx, val_idx
#
#     if mode == "unseen_sentence":
#         holdout_sents = split_info.get("holdout_sentences", None)
#         if holdout_sents is None:
#             frac = float(split_info["holdout_fraction"])
#             rng = np.random.default_rng(seed)
#             unique = sorted(set(sentences))
#             rng.shuffle(unique)
#             n_holdout = max(1, int(len(unique) * frac))
#             holdout_sents = set(unique[:n_holdout])
#         else:
#             holdout_sents = set(holdout_sents)
#
#         val_idx = [i for i, s in enumerate(sentences) if s in holdout_sents]
#         train_idx = [i for i, s in enumerate(sentences) if s not in holdout_sents]
#         return train_idx, val_idx
#
#     if mode == "unseen_subject":
#         holdout_codes = split_info.get("holdout_subjects", None)
#         if holdout_codes is None:
#             raise ValueError("split.json missing holdout_subjects for unseen_subject evaluation.")
#
#         code_to_idx = {c: i for i, c in enumerate(subject_codes)}
#         holdout_ids = {code_to_idx[c] for c in holdout_codes}
#
#         val_idx = [i for i in range(n) if int(subj_ids[i]) in holdout_ids]
#         train_idx = [i for i in range(n) if int(subj_ids[i]) not in holdout_ids]
#         return train_idx, val_idx
#
#     raise ValueError(f"Unknown split_mode in split.json: {mode}")


# -------------------- streaming dataset --------------------

@dataclass(frozen=True)
class TrialRef:
    """Points to one trial inside one run file."""
    sub_code: str
    run_idx: int
    trial_in_run: int
    subject_index: int
    sentence: str
    task: str  # "read" or "imagine"

class ChiscoStreamingEvalDataset(Dataset):
    """
    Streaming dataset for evaluation.
    - Stores only an index of trials (sentences + where to find the EEG).
    - Loads run pickle on-demand in __getitem__.
    - Caches the most recently used run in memory to avoid re-reading for sequential access.
    """

    def __init__(self, root: str, trial_index: List[TrialRef]):
        self.root = root
        self.trial_index = trial_index

        # tiny cache: last run file loaded
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
        # Keep preprocessing consistent with your CHISCO code:
        eeg = arr[0, :122, :] * 1_000_000.0
        eeg = eeg.astype(np.float32)
        eeg = torch.from_numpy(eeg)  # (122, 1651)

        return {
            "eeg": eeg,
            "subject_id": torch.tensor(ref.subject_index, dtype=torch.long),
            "sentence": ref.sentence,
            # Stable trial identifiers for paired model comparisons.
            "sub_code": ref.sub_code,
            "run_idx": int(ref.run_idx),
            "trial_in_run": int(ref.trial_in_run),
        }

def build_streaming_index(
    root: str,
    subject_codes: List[str],
    task: str,
    n_runs_read: int = 45,
    n_runs_imagine: int = 45,
) -> Tuple[List[TrialRef], List[str], torch.Tensor]:
    """
    Build an index without ever stacking EEG into RAM.

    Returns:
      - trial_index: List[TrialRef] (length N)
      - sentences  : List[str]      (length N)  (same order as trial_index)
      - subj_ids   : Long tensor    (length N)  (same order as trial_index)
    """
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




def _batch_value(batch, key: str, i: int):
    """Return a JSON-serializable scalar/list item from a DataLoader-collated batch."""
    value = batch[key]
    if torch.is_tensor(value):
        return value[i].item()
    return value[i]

# -------------------- eval --------------------

@torch.no_grad()
def evaluate_global_retrieval_streaming(
    model,
    dataset: Dataset,
    text_matrix: torch.Tensor,
    sentence_to_id: Dict[str, int],
    device: torch.device,
    batch_size: int = 32,
    topk_max: int = 50,
) -> Dict[str, float]:
    """
    Same metric logic as your original evaluate_global_retrieval(),
    but data comes from a streaming dataset instead of a giant X tensor.
    """
    model.eval()
    text_matrix = text_matrix.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ks = [1, 5, 10, 50]
    top_counts = {k: 0 for k in ks}
    total = 0
    trial_rows: List[Dict[str, Any]] = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        batch_sentences = batch["sentence"]

        eeg_emb = model(eeg, subj)
        eeg_emb = torch.nn.functional.normalize(eeg_emb, dim=1)

        sim = eeg_emb @ text_matrix.T
        k_use = min(topk_max, sim.shape[1])
        _, indices = sim.topk(k=k_use, dim=1)

        for i, sent in enumerate(batch_sentences):
            trial_number = total
            total += 1
            sent = str(sent).strip()
            true_id = sentence_to_id.get(sent, None)

            # Stable identifiers for pairing this trial across model runs.
            sub_code = str(_batch_value(batch, "sub_code", i))
            run_idx = int(_batch_value(batch, "run_idx", i))
            trial_in_run = int(_batch_value(batch, "trial_in_run", i))
            stable_trial_id = f"{sub_code}_run{run_idx:02d}_trial{trial_in_run:04d}"

            if true_id is None:
                # Keep identical aggregate behavior to original: count in total, but skip scoring.
                trial_rows.append({
                    "trial_number": int(trial_number),
                    "trial_id": stable_trial_id,
                    "sub_code": sub_code,
                    "run_idx": run_idx,
                    "trial_in_run": trial_in_run,
                    "sentence": sent,
                    "true_id": None,
                    "rank_within_top50": None,
                    "top1_correct": 0,
                    "top5_correct": 0,
                    "top10_correct": 0,
                    "top50_correct": 0,
                })
                continue

            cand = indices[i].tolist()
            top1_correct = int(true_id == cand[0])
            top5_correct = int(true_id in cand[:5])
            top10_correct = int(true_id in cand[:10])
            top50_correct = int(true_id in cand[:50])

            if top1_correct:
                top_counts[1] += 1
            if top5_correct:
                top_counts[5] += 1
            if top10_correct:
                top_counts[10] += 1
            if top50_correct:
                top_counts[50] += 1

            rank = cand.index(true_id) + 1 if true_id in cand else None

            trial_rows.append({
                "trial_number": int(trial_number),
                "trial_id": stable_trial_id,
                "sub_code": sub_code,
                "run_idx": run_idx,
                "trial_in_run": trial_in_run,
                "sentence": sent,
                "true_id": int(true_id),
                "rank_within_top50": rank,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "top10_correct": top10_correct,
                "top50_correct": top50_correct,
            })

    return {
        "top1": top_counts[1] / total * 100.0,
        "top5": top_counts[5] / total * 100.0,
        "top10": top_counts[10] / total * 100.0,
        "top50": top_counts[50] / total * 100.0,
        "total": float(total),
        "trial_rows": trial_rows,
    }


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate global retrieval for an experiment run folder.")
    ap.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to an experiment run directory containing config.json/split.json/best_checkpoint.pt",
    )
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Optional explicit checkpoint path (overrides exp-dir/best_checkpoint.pt)")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
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
    print("Using device:", device)

    # Pull required fields from config.json (unchanged)
    chisco_root = cfg.get("chisco_root") or cfg.get("CHISCO_ROOT") or "./ds005170/derivatives/preprocessed_pkl"
    sent_emb_path = cfg.get("sent_emb_path") or cfg.get("SENT_EMB_PATH") or "./ds005170/json/sentence_bert_embs.json"
    task = cfg.get("task") or cfg.get("TASK") or "read"

    subject_codes_str = cfg.get("subject_codes")
    if isinstance(subject_codes_str, str):
        subject_codes = [s.strip() for s in subject_codes_str.split(",") if s.strip()]
    elif isinstance(subject_codes_str, list):
        subject_codes = subject_codes_str
    else:
        subject_codes = cfg.get("SUBJECT_CODES") or [f"{i:02d}" for i in range(1, 6)]

    seed = int(cfg.get("seed") or cfg.get("RANDOM_SEED") or split_info.get("seed") or 42)
    val_split = float(cfg.get("val_split") or cfg.get("VAL_SPLIT") or 0.2)
    shuffle = bool(cfg.get("shuffle_data") if "shuffle_data" in cfg else cfg.get("SHUFFLE_DATA", True))

    # 1) Build a streaming index (NO X tensor)
    trial_index, sentences, subj_ids = build_streaming_index(
        root=chisco_root,
        subject_codes=subject_codes,
        task=task,
    )

    n_samples = len(sentences)
    n_channels = 122  # fixed by preprocessing
    n_subjects = len(subject_codes)

    # 2) Recreate exact split (same code path, just using our sentences/subj_ids lists)
    # train_idx, val_idx = split_indices_from_splitinfo(
    #     split_info=split_info,
    #     sentences=sentences,
    #     subj_ids=subj_ids,
    #     subject_codes=subject_codes,
    #     seed=seed,
    #     val_split=val_split,
    #     shuffle=shuffle,
    # )

    train_idx, val_idx, test_idx = split_indices_from_splitinfo(
        split_info=split_info,
        sentences=sentences,
    )

    task_str = task.upper()
    split_mode = split_info.get("split_mode", "iid")
    print("\n=== Global Retrieval Evaluation ===")
    print(f"Task        : {task_str}")
    print(f"Split mode  : {split_mode}")
    print(f"Val trials  : {len(val_idx)}")

    # 3) Load text vocab matrix (unchanged)
    sent2emb = load_sentence_embeddings(sent_emb_path)
    text_matrix, id_to_sentence, sentence_to_id = build_text_embedding_matrix(sent2emb)
    text_dim = int(text_matrix.shape[1])
    print(f"Text vocab size: {len(id_to_sentence)} | text_dim={text_dim}")

    # 4) Build model from checkpoint/config (unchanged)
    checkpoint = torch.load(ckpt_path, map_location=device)

    c2 = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    def _get(name: str, default):
        return c2.get(name, cfg.get(name, default))

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

    arch = str(_get("arch", "brainmodule")).lower()

    if arch == "chisco_baseline":
        baseline_layer = int(_get("baseline_layer", 0))
        baseline_pooling = _get("baseline_pooling", "mean")
        baseline_size1 = int(_get("baseline_size1", 8))
        baseline_size2 = int(_get("baseline_size2", 8))
        baseline_feel1 = int(_get("baseline_feel1", 125))
        baseline_feel2 = int(_get("baseline_feel2", 25))
        baseline_dropout1 = float(_get("baseline_dropout1", 0.1))
        baseline_dropout2 = float(_get("baseline_dropout2", 0.1))
        baseline_latent_dim = int(_get("baseline_latent_dim", 256))

        model = _CHISCOBaselineWrapper(
            encoder=CHISCOBaselineEEGEncoder(
                n_channels=n_channels,
                text_dim=text_dim,
                latent_dim=baseline_latent_dim,
                dropout1=baseline_dropout1,
                dropout2=baseline_dropout2,
                feel1=baseline_feel1,
                feel2=baseline_feel2,
                size1=baseline_size1,
                size2=baseline_size2,
                layer=baseline_layer,
                pooling=baseline_pooling,
            )
        ).to(device)
    else:
        model = EEGToTextEmbedModel(
            n_channels=n_channels,
            n_subjects=n_subjects,
            text_dim=text_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            final_hidden=final_hidden,
            n_blocks=n_blocks,
            pooling=pooling,
            use_frontend=True,
            use_brainmodule=True,
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

    # 5) Streaming val dataset + eval (THIS is the memory fix)
    base_ds = ChiscoStreamingEvalDataset(root=chisco_root, trial_index=trial_index)
    val_ds = Subset(base_ds, test_idx)

    print("\nRunning global retrieval evaluation (streaming, low-RAM)...")
    results = evaluate_global_retrieval_streaming(
        model=model,
        dataset=val_ds,
        text_matrix=text_matrix,
        sentence_to_id=sentence_to_id,
        device=device,
        batch_size=args.eval_batch_size,
    )

    trial_rows = results.pop("trial_rows")

    out_path = os.path.join(exp_dir, "global_retrieval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    trial_out_path = os.path.join(exp_dir, "global_retrieval_trials.json")
    with open(trial_out_path, "w", encoding="utf-8") as f:
        json.dump(trial_rows, f, indent=2, ensure_ascii=False)

    print("\n=== Global Retrieval Accuracy ===")
    print(f"Top-1  : {results['top1']:.2f}%")
    print(f"Top-5  : {results['top5']:.2f}%")
    print(f"Top-10 : {results['top10']:.2f}%")
    print(f"Top-50 : {results['top50']:.2f}%")
    print(f"(Out of {int(results['total'])} validation trials)")
    print(f"Saved summary: {out_path}")
    print(f"Saved trial-level correctness: {trial_out_path}")

if __name__ == "__main__":
    main()

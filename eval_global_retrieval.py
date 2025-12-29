# eval_global_retrieval_from_experiment.py
#
# Evaluate EEG→text embeddings via *global retrieval* using one saved experiment folder.
#
# Usage:
#   python eval_global_retrieval_from_experiment.py --exp-dir experiments/eeg2text/read/iid/<run_id>
#
# What it does:
#   - loads best_checkpoint.pt
#   - loads config.json and split.json from that run folder
#   - reconstructs the exact validation set for:
#       iid / unseen_sentence / unseen_subject
#   - evaluates global retrieval against ALL sentence embeddings in SENT_EMB_PATH
#
# Notes:
#   - Assumes the checkpoint contains model_state_dict and the model architecture params
#     (latent_dim, hidden_channels, etc.). If you have the CLI script, store those in config.json
#     and in checkpoint["config"] too.
#   - If you used few-shot calibration, point --exp-dir to the calibration subfolder that contains
#     checkpoint.pt + split.json, OR pass --ckpt explicitly.

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make ds005170 importable
sys.path.append("./ds005170")

from data_loader import (
    load_chisco_read_all_subjects,
    load_chisco_imagine_all_subjects,
    ChiscoImagineDataset,
)

# IMPORTANT:
# Prefer importing the model class from the *same code version* used in training.
# If you now have a CLI training script, import EEGToTextEmbedModel from there.
# Fallback: import from train_eeg_to_text_embed.py.
try:
    from updated_train_eeg_to_text_embed import EEGToTextEmbedModel  # if you use the CLI script
except Exception:
    from train_eeg_to_text_embed import EEGToTextEmbedModel


# -------------------- helpers --------------------

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sentence_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_text_embedding_matrix(sent2emb: Dict[str, List[float]]) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
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
    Recreate the split based on split.json.
    """
    mode = split_info.get("split_mode", "iid")
    n = len(sentences)

    if mode == "iid":
        _, val_idx = make_train_val_indices(n, val_split=val_split, shuffle=shuffle, seed=seed)
        return list(range(n))[int(n * val_split):], val_idx  # train_idx, val_idx

    if mode == "unseen_sentence":
        # We need the exact holdout set used originally.
        # If you saved holdout sentences in split.json, use them. Otherwise we recreate deterministically.
        holdout_sents = split_info.get("holdout_sentences", None)
        if holdout_sents is None:
            # deterministic recreation from seed + fraction + sorted unique sentences
            frac = float(split_info["holdout_fraction"])
            rng = np.random.default_rng(seed)
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
        # Use exact held-out subject codes/indices saved in split.json
        holdout_codes = split_info.get("holdout_subjects", None)
        if holdout_codes is None:
            raise ValueError("split.json missing holdout_subjects for unseen_subject evaluation.")

        # map "01".."05" -> index used by your pipeline
        code_to_idx = {c: i for i, c in enumerate(subject_codes)}
        holdout_ids = {code_to_idx[c] for c in holdout_codes}

        val_idx = [i for i in range(n) if int(subj_ids[i]) in holdout_ids]
        train_idx = [i for i in range(n) if int(subj_ids[i]) not in holdout_ids]
        return train_idx, val_idx

    raise ValueError(f"Unknown split_mode in split.json: {mode}")

@torch.no_grad()
def evaluate_global_retrieval(
    model,
    X: torch.Tensor,
    sentences: List[str],
    subj_ids: torch.Tensor,
    text_matrix: torch.Tensor,
    sentence_to_id: Dict[str, int],
    device: torch.device,
    batch_size: int = 32,
    topk_max: int = 50,
) -> Dict[str, float]:
    """
    Global retrieval: compare each EEG embedding to ALL sentence embeddings in text_matrix.
    """
    model.eval()
    text_matrix = text_matrix.to(device)

    eeg_dataset = ChiscoImagineDataset(
        eeg=X,
        sentences=sentences,
        subject_ids=subj_ids,
        targets=None,
        augment=False,
    )

    loader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=False)

    ks = [1, 5, 10, 50]
    top_counts = {k: 0 for k in ks}
    total = 0

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
            total += 1
            true_id = sentence_to_id.get(sent, None)
            if true_id is None:
                # should not happen if SENT_EMB_PATH covers all sentences,
                # but we skip to be safe
                continue

            cand = indices[i].tolist()
            if true_id == cand[0]:
                top_counts[1] += 1
            if true_id in cand[:5]:
                top_counts[5] += 1
            if true_id in cand[:10]:
                top_counts[10] += 1
            if true_id in cand[:50]:
                top_counts[50] += 1

    return {
        "top1": top_counts[1] / total * 100.0,
        "top5": top_counts[5] / total * 100.0,
        "top10": top_counts[10] / total * 100.0,
        "top50": top_counts[50] / total * 100.0,
        "total": float(total),
    }


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate global retrieval for an experiment run folder.")
    ap.add_argument("--exp-dir", type=str, required=True,
                    help="Path to an experiment run directory containing config.json/split.json/best_checkpoint.pt")
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

    # Pull required fields from config.json (works for both old and CLI config snapshots)
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

    # 1) Load full dataset for the correct TASK
    if task == "read":
        X, sentences, subj_ids = load_chisco_read_all_subjects(root=chisco_root, subject_codes=subject_codes)
    elif task == "imagine":
        X, sentences, subj_ids = load_chisco_imagine_all_subjects(root=chisco_root, subject_codes=subject_codes)
    else:
        raise ValueError(f"Unknown task in config.json: {task}")

    n_samples, n_channels, _ = X.shape
    n_subjects = len(subject_codes)

    # 2) Recreate exact split
    train_idx, val_idx = split_indices_from_splitinfo(
        split_info=split_info,
        sentences=sentences,
        subj_ids=subj_ids,
        subject_codes=subject_codes,
        seed=seed,
        val_split=val_split,
        shuffle=shuffle,
    )

    X_val = X[val_idx]
    sentences_val = [sentences[i] for i in val_idx]
    subj_ids_val = subj_ids[val_idx]
    task_str = task.upper()
    split_mode = split_info.get("split_mode", "iid")
    print("\n=== Global Retrieval Evaluation ===")
    print(f"Task        : {task_str}")
    print(f"Split mode  : {split_mode}")

    print(f"Split mode: {split_info.get('split_mode')}")
    print(f"Val trials: {len(val_idx)}")

    # 3) Load text vocab matrix
    sent2emb = load_sentence_embeddings(sent_emb_path)
    text_matrix, id_to_sentence, sentence_to_id = build_text_embedding_matrix(sent2emb)
    text_dim = int(text_matrix.shape[1])
    print(f"Text vocab size: {len(id_to_sentence)} | text_dim={text_dim}")

    # 4) Build model from checkpoint/config
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Try to infer architecture from checkpoint["config"] first, then config.json.
    c2 = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    def _get(name: str, default):
        return c2.get(name, cfg.get(name, default))

    # These names match your CLI script flags (lowercase) AND older Config (UPPERCASE).
    latent_dim = int(_get("latent_dim", _get("LATENT_DIM", 256)))
    hidden_channels = int(_get("hidden_channels", _get("HIDDEN_CHANNELS", 256)))
    final_hidden = int(_get("final_hidden", _get("FINAL_HIDDEN", 256)))
    n_blocks = int(_get("n_blocks", _get("N_BLOCKS", 5)))
    pooling = _get("pooling", _get("POOLING", "mean"))

    # Toggles (if not present, default to True to match original behavior)
    use_eegnet = bool(_get("use_eegnet", True))
    eegnet_temporal_kern = int(_get("eegnet_temporal_kern", 64))
    eegnet_f1 = int(_get("eegnet_f1", 16))
    eegnet_d = int(_get("eegnet_d", 2))
    raw_frontend_out_ch = int(_get("raw_frontend_out_ch", 32))

    use_spatial_attn = bool(_get("use_spatial_attn", True))
    use_subject_layer = bool(_get("use_subject_layer", True))
    use_residual_blocks = bool(_get("use_residual_blocks", True))

    # Instantiate model
    model = EEGToTextEmbedModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        text_dim=text_dim,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        final_hidden=final_hidden,
        n_blocks=n_blocks,
        pooling=pooling,
        # toggles (the CLI model signature supports these)
        use_eegnet=use_eegnet,
        eegnet_temporal_kern=eegnet_temporal_kern,
        eegnet_F1=eegnet_f1,
        eegnet_D=eegnet_d,
        use_spatial_attn=use_spatial_attn,
        use_subject_layer=use_subject_layer,
        use_residual_blocks=use_residual_blocks,
        raw_frontend_out_ch=raw_frontend_out_ch,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # 5) Global retrieval on validation set
    print("\nRunning global retrieval evaluation...")
    results = evaluate_global_retrieval(
        model=model,
        X=X_val,
        sentences=sentences_val,
        subj_ids=subj_ids_val,
        text_matrix=text_matrix,
        sentence_to_id=sentence_to_id,
        device=device,
        batch_size=args.eval_batch_size,
    )

    # Save results into the experiment folder
    out_path = os.path.join(exp_dir, "global_retrieval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Global Retrieval Accuracy ===")
    print(f"Top-1  : {results['top1']:.2f}%")
    print(f"Top-5  : {results['top5']:.2f}%")
    print(f"Top-10 : {results['top10']:.2f}%")
    print(f"Top-50 : {results['top50']:.2f}%")
    print(f"(Out of {int(results['total'])} validation trials)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

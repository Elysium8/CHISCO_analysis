# choose_one_testing.py
#
# Two-way forced-choice (2AFC) evaluation.
#
# CHANGE SUMMARY:
# - Validation indices are STRICTLY taken from split.json (train_idx / val_idx only).
# - FIXED model construction: only pass valid model hyperparameters
#   (prevents crash from keys like 'chisco_root').
#
import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append("./ds005170")

from data_loader import load_read_run, load_imagine_run  # type: ignore

try:
    from updated_train_eeg_to_text_embed import EEGToTextEmbedModel
except Exception:
    from train_eeg_to_text_embed import EEGToTextEmbedModel


# -------------------- helpers --------------------

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sentence_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_text_embedding_matrix(
    sent2emb: Dict[str, List[float]]
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    sentences = sorted(sent2emb.keys())
    sentence_to_id = {s: i for i, s in enumerate(sentences)}
    mat = np.stack([np.array(sent2emb[s], dtype=np.float32) for s in sentences])
    mat = torch.tensor(mat)
    mat = torch.nn.functional.normalize(mat, dim=1)
    return mat, sentences, sentence_to_id


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
    Force the use of explicit indices saved in split.json.
    """
    if "train_idx" in split_info and "val_idx" in split_info:
        train_idx = split_info["train_idx"]
        val_idx = split_info["val_idx"]

        if not val_idx:
            raise ValueError("The 'val_idx' list in split.json is empty.")

        return train_idx, val_idx

    mode = split_info.get("split_mode", "unknown")
    raise KeyError(
        f"CRITICAL ERROR: 'train_idx' or 'val_idx' not found in split.json. "
        f"Evaluation for mode '{mode}' cannot proceed safely."
    )


# -------------------- dataset --------------------

@dataclass(frozen=True)
class TrialRef:
    sub_code: str
    run_idx: int
    trial_in_run: int
    subject_index: int
    sentence: str
    task: str

class ChiscoStreamingEvalDataset(Dataset):
    def __init__(self, root: str, trial_index: List[TrialRef]):
        self.root = root
        self.trial_index = trial_index
        self._cache_key = None
        self._cache_trials = None

    def __len__(self):
        return len(self.trial_index)

    def _load_run_trials(self, sub_code: str, run_idx: int, task: str):
        key = (sub_code, run_idx, task)
        if key == self._cache_key and self._cache_trials is not None:
            return self._cache_trials

        trials = (
            load_read_run(self.root, sub_code, run_idx)
            if task == "read"
            else load_imagine_run(self.root, sub_code, run_idx)
        )
        self._cache_key = key
        self._cache_trials = trials
        return trials

    def __getitem__(self, idx):
        ref = self.trial_index[idx]
        trials = self._load_run_trials(ref.sub_code, ref.run_idx, ref.task)
        trial = trials[ref.trial_in_run]

        eeg = trial["input_features"][0, :122, :] * 1e6
        eeg = torch.from_numpy(eeg.astype(np.float32))

        return {
            "eeg": eeg,
            "subject_id": torch.tensor(ref.subject_index, dtype=torch.long),
            "sentence": ref.sentence,
        }


def build_streaming_index(root, subject_codes, task):
    trial_index, sentences, subj_ids = [], [], []

    n_runs = 45 if task == "read" else 45
    loader = load_read_run if task == "read" else load_imagine_run

    for si, sc in enumerate(subject_codes):
        for r in range(1, n_runs + 1):
            trials = loader(root, sc, r)
            if not trials:
                continue
            for ti, tr in enumerate(trials):
                s = str(tr["text"]).strip()
                trial_index.append(TrialRef(sc, r, ti, si, s, task))
                sentences.append(s)
                subj_ids.append(si)

    return trial_index, sentences, torch.tensor(subj_ids)


# -------------------- evaluation --------------------

@torch.no_grad()
def evaluate_2afc(
    model,
    dataset,
    text_matrix,
    sentence_to_id,
    device,
    seed,
    batch_size,
    negative_pool=None,
):
    rng = np.random.default_rng(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    vocab_size = text_matrix.shape[0]
    text_matrix = text_matrix.to(device)

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        sents = batch["sentence"]

        eeg_emb = torch.nn.functional.normalize(model(eeg, subj), dim=1)

        for i, s in enumerate(sents):
            tid = sentence_to_id.get(s)
            if tid is None:
                continue

            # sample random negative
            fid = int(rng.integers(0, vocab_size - 1))
            if fid >= tid:
                fid += 1

            sim_t = (eeg_emb[i] * text_matrix[tid]).sum()
            sim_f = (eeg_emb[i] * text_matrix[fid]).sum()

            correct += int(sim_t > sim_f)
            total += 1

    return 100.0 * correct / total


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = read_json(os.path.join(args.exp_dir, "config.json"))
    split = read_json(os.path.join(args.exp_dir, "split.json"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    subject_codes = cfg.get("subject_codes")
    if isinstance(subject_codes, str):
        subject_codes = [s.strip() for s in subject_codes.split(",") if s.strip()]

    task = cfg.get("task", "read")
    root = cfg["chisco_root"]
    sent_emb_path = cfg["sent_emb_path"]

    trial_index, sentences, subj_ids = build_streaming_index(root, subject_codes, task)

    _, val_idx = split_indices_from_splitinfo(
        split_info=split,
        sentences=sentences,
        subj_ids=subj_ids,
        subject_codes=subject_codes,
        seed=0,
        val_split=0.0,
        shuffle=False,
    )

    val_ds = Subset(ChiscoStreamingEvalDataset(root, trial_index), val_idx)

    sent2emb = load_sentence_embeddings(sent_emb_path)
    text_matrix, _, sentence_to_id = build_text_embedding_matrix(sent2emb)

    ckpt = torch.load(os.path.join(args.exp_dir, "best_checkpoint.pt"), map_location=device)

    # -------- FIXED MODEL CONSTRUCTION --------
    c = ckpt.get("config", {})

    def get(name, default):
        return c.get(name, cfg.get(name, default))

    model = EEGToTextEmbedModel(
        n_channels=122,
        n_subjects=len(subject_codes),
        text_dim=text_matrix.shape[1],
        latent_dim=int(get("latent_dim", 256)),
        hidden_channels=int(get("hidden_channels", 256)),
        final_hidden=int(get("final_hidden", 256)),
        n_blocks=int(get("n_blocks", 5)),
        pooling=get("pooling", "mean"),
        use_eegnet=bool(get("use_eegnet", True)),
        eegnet_temporal_kern=int(get("eegnet_temporal_kern", 64)),
        eegnet_F1=int(get("eegnet_f1", 16)),
        eegnet_D=int(get("eegnet_d", 2)),
        raw_frontend_out_ch=int(get("raw_frontend_out_ch", 32)),
        use_spatial_attn=bool(get("use_spatial_attn", True)),
        use_subject_layer=bool(get("use_subject_layer", True)),
        use_residual_blocks=bool(get("use_residual_blocks", True)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    acc = evaluate_2afc(
        model,
        val_ds,
        text_matrix,
        sentence_to_id,
        device,
        seed=123,
        batch_size=args.eval_batch_size,
    )

    print(f"\n2AFC accuracy: {acc:.2f} %")


if __name__ == "__main__":
    main()

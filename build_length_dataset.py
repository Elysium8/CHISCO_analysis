import json
from typing import Dict, List, Tuple

import torch

from data_loader import load_chisco_read_all_subjects


def load_sentence_length_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)  # {sentence: length_int}
    return {k: int(v) for k, v in m.items()}


def build_sentence_length_tensors(
    chisco_root: str,
    subject_codes: List[str],
    length_json_path: str,
):
    """
    Returns:
      X:         (N, 122, 1651) float32
      lengths:   (N,) float32   (character length)
      sentences: list[str]
      subj_ids:  (N,) long      (0..n_subjects-1)
    """
    X, sentences, subj_ids = load_chisco_read_all_subjects(
        root=chisco_root,
        subject_codes=subject_codes,
    )

    sent2len = load_sentence_length_map(length_json_path)

    kept_X = []
    kept_len = []
    kept_sent = []
    kept_subj = []

    missing = 0
    for i, s in enumerate(sentences):
        if s not in sent2len:
            missing += 1
            continue
        kept_X.append(X[i])
        kept_len.append(float(sent2len[s]))
        kept_sent.append(s)
        kept_subj.append(subj_ids[i])

    if len(kept_X) == 0:
        raise RuntimeError("No samples matched sentence lengths. Check JSON paths / keys.")

    X_kept = torch.stack(kept_X, dim=0)
    y_kept = torch.tensor(kept_len, dtype=torch.float32)
    subj_kept = torch.stack(kept_subj, dim=0)

    print(f"Matched {len(kept_X)} / {len(sentences)} trials (missing={missing}).")
    return X_kept, y_kept, kept_sent, subj_kept

import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- Low-level: load one run for one subject ----------

def load_imagine_run(
    root: str,
    sub_code: str,
    run_idx: int,
) -> List[dict]:
    """
    Load a single 'imagine' run for one subject.

    Args:
        root:     path to 'preprocessed_pkl' directory
                  e.g. "./Chisco/derivatives/preprocessed_pkl"
        sub_code: subject code WITHOUT 'sub-' prefix, e.g. "01", "02"
        run_idx:  run index, starting at 1

    Returns:
        List of trial dicts (possibly empty if file missing).
    """
    run_str = f"{run_idx:03d}"  # 1 -> "01"
    fname = f"sub-{sub_code}_task-imagine_run-{run_str}_eeg.pkl"
    path = os.path.join(root, f"sub-{sub_code}", "eeg", fname)

    if not os.path.exists(path):
        # Some subjects might have fewer runs
        return []

    with open(path, "rb") as f:
        trials = pickle.load(f)

    return trials


# ---------- Mid-level: load all runs for one subject ----------

def load_subject_imagine_trials(
    root: str,
    sub_code: str,
    subject_index: int,
    n_runs: int = 45,
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Load all imagine trials for a single subject and return:

        eeg_list      : list[Tensor]  each (C=122, T=1651)
        sentence_list : list[str]
        subject_ids   : list[int]    all == subject_index

    We follow the CHISCO preprocessing: take channels [:122], scale by 1e6,
    cast to float32.
    """
    eeg_list: List[torch.Tensor] = []
    sentence_list: List[str] = []
    subject_ids: List[int] = []

    for run_idx in range(1, n_runs + 1):
        trials = load_imagine_run(root, sub_code, run_idx)
        for trial in trials:
            arr = trial["input_features"]  # (1, 125, 1651), float64
            assert isinstance(arr, np.ndarray)
            assert arr.shape[0] == 1 and arr.ndim == 3

            # (1, 125, 1651) -> (122, 1651) and scale
            eeg = arr[0, :122, :] * 1_000_000.0  # same scaling as CHISCO code


            # trialwise z-score per channel (maybe, should test)
            # mean = eeg.mean(axis=1, keepdims=True)
            # std = eeg.std(axis=1, keepdims=True) + 1e-6
            # eeg = (eeg - mean) / std

            eeg = eeg.astype(np.float32)
            eeg = torch.from_numpy(eeg)          # (122, 1651)

            sentence = trial["text"].strip()

            eeg_list.append(eeg)
            sentence_list.append(sentence)
            subject_ids.append(subject_index)

    return eeg_list, sentence_list, subject_ids


# ---------- High-level: load multiple subjects into arrays ----------

def load_chisco_imagine_all_subjects(
    root: str,
    subject_codes: Sequence[str],
    n_runs: int = 45,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Load imagine EEG for all subjects into big tensors.

    Args:
        root:          preprocessed_pkl root directory.
        subject_codes: list like ["01", "02", "03", ...].
        n_runs:        maximum number of runs per subject to try.

    Returns:
        X:          (N_trials, C=122, T=1651) float32 tensor
        sentences:  list[str] length N_trials
        subject_ids:(N_trials,) long tensor in [0, n_subjects-1]
    """
    all_eeg: List[torch.Tensor] = []
    all_sentences: List[str] = []
    all_sub_ids: List[int] = []

    for subj_idx, sub_code in enumerate(subject_codes):
        eeg_list, sent_list, sub_ids = load_subject_imagine_trials(
            root=root,
            sub_code=sub_code,
            subject_index=subj_idx,
            n_runs=n_runs,
        )
        all_eeg.extend(eeg_list)
        all_sentences.extend(sent_list)
        all_sub_ids.extend(sub_ids)

    X = torch.stack(all_eeg, dim=0)                   # (N, 122, 1651)
    subject_ids = torch.tensor(all_sub_ids, dtype=torch.long)  # (N,)

    return X, all_sentences, subject_ids



# ---------- Low-level: load one READ run for one subject ----------

def load_read_run(
    root: str,
    sub_code: str,
    run_idx: int,
) -> List[dict]:
    """
    Load a single 'read' run for one subject.

    Args:
        root:     path to 'preprocessed_pkl' directory
                  e.g. "./ds005170/derivatives/preprocessed_pkl"
        sub_code: subject code WITHOUT 'sub-' prefix, e.g. "01", "02"
        run_idx:  run index, starting at 1

    Returns:
        List of trial dicts (possibly empty if file missing).
    """
    run_str = f"{run_idx:02d}"
    fname = f"sub-{sub_code}_task-read_run-{run_str}_eeg.pkl"
    path = os.path.join(root, f"sub-{sub_code}", "eeg", fname)

    if not os.path.exists(path):
        # Some subjects might have fewer read runs
        return []

    with open(path, "rb") as f:
        trials = pickle.load(f)

    return trials


# ---------- Mid-level: load all READ runs for one subject ----------

def load_subject_read_trials(
    root: str,
    sub_code: str,
    subject_index: int,
    n_runs: int = 25,
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Load all READ trials for a single subject.

    Returns:
        eeg_list      : list[Tensor] each (C=122, T=1651)
        sentence_list : list[str]
        subject_ids   : list[int] all == subject_index
    """
    eeg_list: List[torch.Tensor] = []
    sentence_list: List[str] = []
    subject_ids: List[int] = []

    for run_idx in range(1, n_runs + 1):
        trials = load_read_run(root, sub_code, run_idx)
        for trial in trials:
            arr = trial["input_features"]  # (1, 125, 1651), float64
            assert isinstance(arr, np.ndarray)
            assert arr.shape[0] == 1 and arr.ndim == 3

            # (1, 125, 1651) -> (122, 1651) and scale to microvolts
            eeg = arr[0, :122, :] * 1_000_000.0
            eeg = eeg.astype(np.float32)
            eeg = torch.from_numpy(eeg)          # (122, 1651)

            sentence = trial["text"].strip()

            eeg_list.append(eeg)
            sentence_list.append(sentence)
            subject_ids.append(subject_index)

    return eeg_list, sentence_list, subject_ids


# ---------- High-level: load READ for all subjects ----------

def load_chisco_read_all_subjects(
    root: str,
    subject_codes: Sequence[str],
    n_runs: int = 25,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Load READ EEG for all subjects into big tensors.

    Args:
        root:          preprocessed_pkl root directory.
        subject_codes: list like ["01", "02", "03", ...].
        n_runs:        max number of read runs per subject to try.

    Returns:
        X:          (N_trials, C=122, T=1651) float32 tensor
        sentences:  list[str] length N_trials
        subject_ids:(N_trials,) long tensor in [0, n_subjects-1]
    """
    all_eeg: List[torch.Tensor] = []
    all_sentences: List[str] = []
    all_sub_ids: List[int] = []

    for subj_idx, sub_code in enumerate(subject_codes):
        eeg_list, sent_list, sub_ids = load_subject_read_trials(
            root=root,
            sub_code=sub_code,
            subject_index=subj_idx,
            n_runs=n_runs,
        )
        all_eeg.extend(eeg_list)
        all_sentences.extend(sent_list)
        all_sub_ids.extend(sub_ids)

    if len(all_eeg) == 0:
        raise RuntimeError(
            f"No READ trials found. Check CHISCO_ROOT and subject codes. "
            f"root={root}, subjects={subject_codes}"
        )

    X = torch.stack(all_eeg, dim=0)                   # (N, 122, 1651)
    subject_ids = torch.tensor(all_sub_ids, dtype=torch.long)  # (N,)

    return X, all_sentences, subject_ids







# ---------- Dataset: labels are flexible ----------
class ChiscoImagineDataset(Dataset):
    """
    Generic CHISCO imagined-speech dataset.

    Always provides:
        - eeg        : Tensor (C, T)
        - subject_id : int
        - sentence   : str

    Optionally, if you pass 'targets', also provides:
        - target     : whatever you like (int, Tensor, etc.)

    This makes it easy to:
        * Genre classification:    targets = integer genre IDs per trial
        * Sentence classification: targets = token IDs, embeddings, etc.
    """
    def __init__(
        self,
        eeg: torch.Tensor,
        sentences: Sequence[str],
        subject_ids: torch.Tensor,
        targets: Optional[Sequence] = None,
        augment: bool = False,          # NEW: turn augmentations on/off
        noise_std: float = 0.01,        # NEW: Gaussian noise std
        jitter_max: int = 10,           # NEW: max temporal shift (samples)
        channel_dropout_p: float = 0.05 # NEW: fraction of channels to drop
    ):
        """
        Args:
            eeg:         (N, C, T) float32 tensor
            sentences:   list of N strings
            subject_ids: (N,) long tensor
            targets:     optional sequence of length N
                         (e.g., genre IDs or sentence encodings)
        """
        assert eeg.ndim == 3, "eeg must be (N, C, T)"
        assert len(sentences) == eeg.shape[0]
        assert subject_ids.shape[0] == eeg.shape[0]
        if targets is not None:
            assert len(targets) == eeg.shape[0]

        self.eeg = eeg
        self.sentences = list(sentences)
        self.subject_ids = subject_ids
        self.targets = targets

        # augmentation params
        self.augment = augment
        self.noise_std = noise_std
        self.jitter_max = jitter_max
        self.channel_dropout_p = channel_dropout_p

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, idx: int):
        # clone so we don't modify the stored tensor in-place
        eeg = self.eeg[idx].clone()              # (C, T)
        subj = self.subject_ids[idx]
        sent = self.sentences[idx]

        # -----------------------------
        #       DATA AUGMENTATION
        # -----------------------------
        if self.augment:
            # 1) Gaussian channel noise
            if self.noise_std > 0 and torch.rand(1) < 0.5:
                eeg = eeg + self.noise_std * torch.randn_like(eeg)

            # 2) Temporal jitter (roll along time)
            if self.jitter_max > 0 and torch.rand(1) < 0.5:
                shift = int(torch.randint(
                    -self.jitter_max, self.jitter_max + 1, (1,)
                ).item())
                if shift != 0:
                    eeg = torch.roll(eeg, shifts=shift, dims=-1)

            # 3) Channel dropout
            if self.channel_dropout_p > 0 and torch.rand(1) < 0.5:
                C = eeg.shape[0]
                drop_mask = torch.rand(C) < self.channel_dropout_p
                if drop_mask.any():
                    eeg[drop_mask, :] = 0.0

        sample = {
            "eeg": eeg,              # (C, T) (possibly augmented)
            "subject_id": subj,      # scalar long
            "sentence": sent,        # str
        }
        if self.targets is not None:
            sample["target"] = self.targets[idx]

        return sample

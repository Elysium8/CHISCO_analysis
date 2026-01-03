import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Low-level: robust run file loading (padding mismatch safe)
# ============================================================

def _find_existing_path(candidates: Sequence[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_imagine_run(root: str, sub_code: str, run_idx: int) -> List[dict]:
    """
    Load a single imagine run for one subject.
    Tries run padding formats: 001, 01, 1.
    """
    candidates = []
    for fmt in (f"{run_idx:03d}", f"{run_idx:02d}", str(run_idx)):
        fname = f"sub-{sub_code}_task-imagine_run-{fmt}_eeg.pkl"
        candidates.append(os.path.join(root, f"sub-{sub_code}", "eeg", fname))

    path = _find_existing_path(candidates)
    if path is None:
        return []

    with open(path, "rb") as f:
        return pickle.load(f)


def load_read_run(root: str, sub_code: str, run_idx: int) -> List[dict]:
    """
    Load a single read run for one subject.
    Tries run padding formats: 001, 01, 1.
    """
    candidates = []
    for fmt in (f"{run_idx:03d}", f"{run_idx:02d}", str(run_idx)):
        fname = f"sub-{sub_code}_task-read_run-{fmt}_eeg.pkl"
        candidates.append(os.path.join(root, f"sub-{sub_code}", "eeg", fname))

    path = _find_existing_path(candidates)
    if path is None:
        return []

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# In-memory loaders (kept for compatibility; RAM-heavy)
# ============================================================

def load_subject_imagine_trials(
    root: str,
    sub_code: str,
    subject_index: int,
    n_runs: int = 45,
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Load all imagine trials for a single subject.

    Returns:
        eeg_list      : list[Tensor] each (122, 1651) float32
        sentence_list : list[str]
        subject_ids   : list[int]
    """
    eeg_list: List[torch.Tensor] = []
    sentence_list: List[str] = []
    subject_ids: List[int] = []

    for run_idx in range(1, n_runs + 1):
        trials = load_imagine_run(root, sub_code, run_idx)
        for trial in trials:
            arr = trial["input_features"]  # (1, 125, 1651)
            eeg = arr[0, :122, :] * 1_000_000.0
            eeg = eeg.astype(np.float32, copy=False)
            eeg = torch.from_numpy(eeg)

            sentence = (trial.get("text") or trial.get("sentence") or "").strip()
            if not sentence:
                continue

            eeg_list.append(eeg)
            sentence_list.append(sentence)
            subject_ids.append(subject_index)

    return eeg_list, sentence_list, subject_ids


def load_subject_read_trials(
    root: str,
    sub_code: str,
    subject_index: int,
    n_runs: int = 45,
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    eeg_list: List[torch.Tensor] = []
    sentence_list: List[str] = []
    subject_ids: List[int] = []

    for run_idx in range(1, n_runs + 1):
        trials = load_read_run(root, sub_code, run_idx)
        for trial in trials:
            arr = trial["input_features"]  # (1, 125, 1651)
            eeg = arr[0, :122, :] * 1_000_000.0
            eeg = eeg.astype(np.float32, copy=False)
            eeg = torch.from_numpy(eeg)

            sentence = (trial.get("text") or trial.get("sentence") or "").strip()
            if not sentence:
                continue

            eeg_list.append(eeg)
            sentence_list.append(sentence)
            subject_ids.append(subject_index)

    return eeg_list, sentence_list, subject_ids


def load_chisco_imagine_all_subjects(
    root: str,
    subject_codes: Sequence[str],
    n_runs: int = 45,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
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

    X = torch.stack(all_eeg, dim=0)
    subject_ids = torch.tensor(all_sub_ids, dtype=torch.long)
    return X, all_sentences, subject_ids


def load_chisco_read_all_subjects(
    root: str,
    subject_codes: Sequence[str],
    n_runs: int = 45,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
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
        raise RuntimeError(f"No READ trials found. root={root}, subjects={subject_codes}")

    X = torch.stack(all_eeg, dim=0)
    subject_ids = torch.tensor(all_sub_ids, dtype=torch.long)
    return X, all_sentences, subject_ids


class ChiscoImagineDataset(Dataset):
    """
    In-memory dataset (unchanged). Kept for compatibility.
    """
    def __init__(
        self,
        eeg: torch.Tensor,
        sentences: Sequence[str],
        subject_ids: torch.Tensor,
        targets: Optional[Sequence] = None,
        augment: bool = False,
        noise_std: float = 0.01,
        jitter_max: int = 10,
        channel_dropout_p: float = 0.05,
    ):
        assert eeg.ndim == 3, "eeg must be (N, C, T)"
        assert len(sentences) == eeg.shape[0]
        assert subject_ids.shape[0] == eeg.shape[0]
        if targets is not None:
            assert len(targets) == eeg.shape[0]

        self.eeg = eeg
        self.sentences = list(sentences)
        self.subject_ids = subject_ids
        self.targets = targets

        self.augment = augment
        self.noise_std = noise_std
        self.jitter_max = jitter_max
        self.channel_dropout_p = channel_dropout_p

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, idx: int):
        eeg = self.eeg[idx].clone()
        subj = self.subject_ids[idx]
        sent = self.sentences[idx]

        if self.augment:
            if self.noise_std > 0 and torch.rand(1) < 0.5:
                eeg = eeg + self.noise_std * torch.randn_like(eeg)

            if self.jitter_max > 0 and torch.rand(1) < 0.5:
                shift = int(torch.randint(-self.jitter_max, self.jitter_max + 1, (1,)).item())
                if shift != 0:
                    eeg = torch.roll(eeg, shifts=shift, dims=-1)

            if self.channel_dropout_p > 0 and torch.rand(1) < 0.5:
                C = eeg.shape[0]
                drop_mask = torch.rand(C) < self.channel_dropout_p
                if drop_mask.any():
                    eeg[drop_mask, :] = 0.0

        sample = {"eeg": eeg, "subject_id": subj, "sentence": sent}
        if self.targets is not None:
            sample["target"] = self.targets[idx]
        return sample


# ============================================================
# STREAMING / ON-DISK Dataset (Option 3)
# ============================================================

@dataclass(frozen=True)
class _TrialRef:
    sub_code: str
    subject_id: int
    run_idx: int
    trial_idx: int
    sentence: str


class _RunCache:
    """
    Very small cache of loaded run trial lists.
    Keeps peak memory bounded while reducing disk I/O.
    """
    def __init__(self, cache_size: int = 2):
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[Tuple[str,int], List[dict]]" = OrderedDict()

    def get(self, key: Tuple[str, int]) -> Optional[List[dict]]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Tuple[str, int], value: List[dict]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)


class ChiscoStreamingDataset(Dataset):
    """
    Streaming dataset that mirrors the sample dict keys your training loop expects:
      - "eeg"        : FloatTensor (122, 1651)
      - "subject_id" : LongTensor scalar
      - "target"     : FloatTensor (D,)
      - "sentence"   : str (kept for split + debugging)

    Also exposes:
      - self.sentences (List[str]) aligned with indices
      - self.subject_ids (torch.LongTensor) aligned with indices

    This dataset builds a lightweight index once (sentences + refs),
    then loads only the required run file on demand in __getitem__.
    """

    def __init__(
        self,
        root: str,
        task: str,
        subject_codes: Sequence[str],
        sent2emb: Dict[str, np.ndarray],
        cache_runs: int = 2,
        n_runs_imagine: int = 45,
        n_runs_read: int = 45,
        n_channels_keep: int = 122,
        scale_to_microvolts: bool = True,
        # augmentation options (same behavior as your in-memory dataset)
        augment: bool = False,
        noise_std: float = 0.01,
        jitter_max: int = 10,
        channel_dropout_p: float = 0.05,
    ):
        super().__init__()
        if task not in ("read", "imagine"):
            raise ValueError("task must be 'read' or 'imagine'")

        self.root = root
        self.task = task
        self.subject_codes = list(subject_codes)
        self.sent2emb = sent2emb

        self.cache = _RunCache(cache_size=cache_runs)

        self.n_runs_imagine = int(n_runs_imagine)
        self.n_runs_read = int(n_runs_read)
        self.n_channels_keep = int(n_channels_keep)
        self.scale_to_microvolts = bool(scale_to_microvolts)

        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.jitter_max = int(jitter_max)
        self.channel_dropout_p = float(channel_dropout_p)

        self.index: List[_TrialRef] = []
        self.sentences: List[str] = []
        self._subject_ids_list: List[int] = []

        self._build_index()
        self.subject_ids = torch.tensor(self._subject_ids_list, dtype=torch.long)

    def _build_index(self) -> None:
        print(f"Building streaming index (task={self.task})...")
        missing_files = 0
        missing_emb = 0
        seen = 0

        for subject_id, sub_code in enumerate(self.subject_codes):
            n_runs = self.n_runs_imagine if self.task == "imagine" else self.n_runs_read
            for run_idx in range(1, n_runs + 1):
                trials = load_imagine_run(self.root, sub_code, run_idx) if self.task == "imagine" else load_read_run(self.root, sub_code, run_idx)
                if len(trials) == 0:
                    missing_files += 1
                    continue

                for trial_idx, trial in enumerate(trials):
                    seen += 1
                    sent = (trial.get("text") or trial.get("sentence") or "").strip()
                    if not sent:
                        continue
                    if self.sent2emb and (sent not in self.sent2emb):
                        missing_emb += 1
                        continue

                    self.index.append(_TrialRef(sub_code, subject_id, run_idx, trial_idx, sent))
                    self.sentences.append(sent)
                    self._subject_ids_list.append(subject_id)

        print(f"Index built: {len(self.index)} usable trials (seen={seen}, missing_run_files={missing_files}, missing_emb={missing_emb})")

    def __len__(self) -> int:
        return len(self.index)

    def _load_run_trials(self, sub_code: str, run_idx: int) -> List[dict]:
        key = (sub_code, run_idx)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        trials = load_imagine_run(self.root, sub_code, run_idx) if self.task == "imagine" else load_read_run(self.root, sub_code, run_idx)
        self.cache.put(key, trials)
        return trials

    def _augment_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return eeg

        eeg = eeg.clone()

        if self.noise_std > 0 and torch.rand(1) < 0.5:
            eeg = eeg + self.noise_std * torch.randn_like(eeg)

        if self.jitter_max > 0 and torch.rand(1) < 0.5:
            shift = int(torch.randint(-self.jitter_max, self.jitter_max + 1, (1,)).item())
            if shift != 0:
                eeg = torch.roll(eeg, shifts=shift, dims=-1)

        if self.channel_dropout_p > 0 and torch.rand(1) < 0.5:
            C = eeg.shape[0]
            drop_mask = torch.rand(C) < self.channel_dropout_p
            if drop_mask.any():
                eeg[drop_mask, :] = 0.0

        return eeg

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self.index[idx]
        trials = self._load_run_trials(ref.sub_code, ref.run_idx)
        trial = trials[ref.trial_idx]

        arr = trial["input_features"]  # (1, 125, 1651)
        eeg = arr[0, : self.n_channels_keep, :]

        if self.scale_to_microvolts:
            eeg = eeg * 1_000_000.0

        # float32 here is the key memory fix
        eeg = np.asarray(eeg, dtype=np.float32)
        eeg_t = torch.from_numpy(eeg)

        eeg_t = self._augment_eeg(eeg_t)

        target_np = self.sent2emb[ref.sentence]  # np.float32 (D,)
        target = torch.from_numpy(target_np)

        return {
            "eeg": eeg_t,
            "subject_id": torch.tensor(ref.subject_id, dtype=torch.long),
            "sentence": ref.sentence,
            "target": target,
        }

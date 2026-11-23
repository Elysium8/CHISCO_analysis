"""Loading CHISCO EEG trials from the preprocessed pickle files.

Each run file ``sub-XX/eeg/sub-XX_task-<task>_run-NNN_eeg.pkl`` holds a list of
trials, where each trial is a dict with ``input_features`` of shape
(1, 125, 1651) in volts and the sentence in ``text``. We keep the first 122
channels (the EEG electrodes) and convert to microvolts.
"""

import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

N_CHANNELS = 122
N_RUNS = 45
TASKS = ("read", "imagine")


def load_run(root: str, sub_code: str, run_idx: int, task: str) -> List[dict]:
    """Load the trials of one run, or return [] if the run file does not exist."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    for run_str in (f"{run_idx:03d}", f"{run_idx:02d}", str(run_idx)):
        path = os.path.join(root, f"sub-{sub_code}", "eeg", f"sub-{sub_code}_task-{task}_run-{run_str}_eeg.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return []


def trial_eeg(trial: dict) -> torch.Tensor:
    """Return a trial's EEG as a float32 tensor of shape (122, T) in microvolts."""
    eeg = trial["input_features"][0, :N_CHANNELS, :] * 1e6
    return torch.from_numpy(np.asarray(eeg, dtype=np.float32))


def trial_sentence(trial: dict) -> str:
    return (trial.get("text") or "").strip()


def load_all_trials(
    root: str,
    subject_codes: Sequence[str],
    task: str,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Load every trial of a task into memory.

    Returns:
        eeg:         (N, 122, T) float32
        sentences:   N sentence strings
        subject_ids: (N,) index of the subject in ``subject_codes``
    """
    eeg, sentences, subject_ids = [], [], []
    for subject_id, sub_code in enumerate(subject_codes):
        for run_idx in range(1, N_RUNS + 1):
            for trial in load_run(root, sub_code, run_idx, task):
                sentence = trial_sentence(trial)
                if not sentence:
                    continue
                eeg.append(trial_eeg(trial))
                sentences.append(sentence)
                subject_ids.append(subject_id)

    if not eeg:
        raise RuntimeError(f"No {task} trials found under {root} for subjects {list(subject_codes)}")

    return torch.stack(eeg), sentences, torch.tensor(subject_ids, dtype=torch.long)


def augment_eeg(eeg: torch.Tensor, noise_std: float, jitter_max: int, channel_dropout_p: float) -> torch.Tensor:
    """Each augmentation is applied independently with probability 0.5."""
    eeg = eeg.clone()

    if noise_std > 0 and torch.rand(1) < 0.5:
        eeg = eeg + noise_std * torch.randn_like(eeg)

    if jitter_max > 0 and torch.rand(1) < 0.5:
        shift = int(torch.randint(-jitter_max, jitter_max + 1, (1,)))
        eeg = torch.roll(eeg, shifts=shift, dims=-1)

    if channel_dropout_p > 0 and torch.rand(1) < 0.5:
        drop = torch.rand(eeg.shape[0]) < channel_dropout_p
        eeg[drop, :] = 0.0

    return eeg


class EEGTextDataset(Dataset):
    """In-memory EEG trials paired with sentence-embedding targets (used for training)."""

    def __init__(
        self,
        eeg: torch.Tensor,
        sentences: Sequence[str],
        subject_ids: torch.Tensor,
        targets: torch.Tensor,
        augment: bool = False,
        noise_std: float = 0.01,
        jitter_max: int = 10,
        channel_dropout_p: float = 0.05,
    ):
        assert len(eeg) == len(sentences) == len(subject_ids) == len(targets)
        self.eeg = eeg
        self.sentences = list(sentences)
        self.subject_ids = subject_ids
        self.targets = targets
        self.augment = augment
        self.noise_std = noise_std
        self.jitter_max = jitter_max
        self.channel_dropout_p = channel_dropout_p

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int) -> dict:
        eeg = self.eeg[idx]
        if self.augment:
            eeg = augment_eeg(eeg, self.noise_std, self.jitter_max, self.channel_dropout_p)
        return {
            "eeg": eeg,
            "subject_id": self.subject_ids[idx],
            "sentence": self.sentences[idx],
            "target": self.targets[idx],
        }


@dataclass(frozen=True)
class TrialRef:
    sub_code: str
    subject_id: int
    run_idx: int
    trial_in_run: int
    sentence: str

    @property
    def trial_id(self) -> str:
        return f"{self.sub_code}_run{self.run_idx:02d}_trial{self.trial_in_run:04d}"


class StreamingEEGDataset(Dataset):
    """Keeps only an index of trials in memory and reads run files on demand.

    Used by the evaluation and analysis scripts, which only need a subset of
    trials (usually the test split) and should not load the full dataset.

    Args:
        sentences: if given, only trials whose sentence is in this collection are kept.
        sent2emb:  if given, each sample also contains its sentence embedding as ``target``.
    """

    def __init__(
        self,
        root: str,
        task: str,
        subject_codes: Sequence[str],
        sentences: Optional[Sequence[str]] = None,
        sent2emb: Optional[Dict[str, Sequence[float]]] = None,
        cache_runs: int = 2,
    ):
        self.root = root
        self.task = task
        self.sent2emb = sent2emb
        self.cache_runs = cache_runs
        self._cache: "OrderedDict[Tuple[str, int], List[dict]]" = OrderedDict()

        keep = None if sentences is None else {s.strip() for s in sentences}

        self.refs: List[TrialRef] = []
        for subject_id, sub_code in enumerate(subject_codes):
            for run_idx in range(1, N_RUNS + 1):
                for trial_in_run, trial in enumerate(load_run(root, sub_code, run_idx, task)):
                    sentence = trial_sentence(trial)
                    if not sentence or (keep is not None and sentence not in keep):
                        continue
                    if sent2emb is not None and sentence not in sent2emb:
                        continue
                    self.refs.append(TrialRef(sub_code, subject_id, run_idx, trial_in_run, sentence))

        self.sentences = [r.sentence for r in self.refs]

    def __len__(self) -> int:
        return len(self.refs)

    def _run_trials(self, sub_code: str, run_idx: int) -> List[dict]:
        key = (sub_code, run_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = load_run(self.root, sub_code, run_idx, self.task)
            if len(self._cache) > self.cache_runs:
                self._cache.popitem(last=False)
        return self._cache[key]

    def __getitem__(self, idx: int) -> dict:
        ref = self.refs[idx]
        trial = self._run_trials(ref.sub_code, ref.run_idx)[ref.trial_in_run]
        sample = {
            "eeg": trial_eeg(trial),
            "subject_id": torch.tensor(ref.subject_id, dtype=torch.long),
            "sentence": ref.sentence,
        }
        if self.sent2emb is not None:
            sample["target"] = torch.as_tensor(self.sent2emb[ref.sentence], dtype=torch.float32)
        return sample

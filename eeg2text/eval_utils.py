"""Helpers shared by the evaluation and analysis scripts."""

import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eeg2text.data import N_CHANNELS, StreamingEEGDataset
from eeg2text.model import build_model

DEFAULT_CHISCO_ROOT = "./ds005170/derivatives/preprocessed_pkl"
DEFAULT_SENT_EMB_PATH = "./ds005170/json/sentence_bert_embs.json"


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "cuda" and torch.cuda.is_available() else "cpu")


def find_run_dirs(root: str, checkpoint_name: str = "best_checkpoint.pt") -> List[str]:
    """All folders under ``root`` that look like a training run."""
    required = {"config.json", "split.json", checkpoint_name}
    return sorted(dirpath for dirpath, _, files in os.walk(root) if required.issubset(files))


def mean_ci95(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and half-width of a normal-approximation 95% CI (1.96 * SEM)."""
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, 0.0
    return mean, 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def load_text_embeddings(path: str) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    """Load the sentence-embedding file as a candidate matrix.

    Returns:
        matrix:         (N_sentences, D), L2-normalised rows, sorted by sentence
        sentences:      row index -> sentence
        sentence_to_id: sentence -> row index
    """
    sent2emb = read_json(path)
    sentences = sorted(sent2emb)
    matrix = torch.tensor(np.stack([np.asarray(sent2emb[s], dtype=np.float32) for s in sentences]))
    matrix = nn.functional.normalize(matrix, dim=1)
    return matrix, sentences, {s: i for i, s in enumerate(sentences)}


class Run:
    """A trained run folder: its config, sentence split and model."""

    def __init__(self, run_dir: str, device: torch.device, checkpoint_path: str = None):
        self.run_dir = run_dir
        self.config = read_json(os.path.join(run_dir, "config.json"))
        self.split = read_json(os.path.join(run_dir, "split.json"))
        self.task = self.config["task"]
        self.subject_codes = self.config["subject_codes"]
        self.chisco_root = self.config.get("chisco_root", DEFAULT_CHISCO_ROOT)
        self.sent_emb_path = self.config.get("sent_emb_path", DEFAULT_SENT_EMB_PATH)
        self.checkpoint_path = checkpoint_path or os.path.join(run_dir, "best_checkpoint.pt")
        self.device = device

    def load_model(self, text_dim: int) -> nn.Module:
        model = build_model(self.config, N_CHANNELS, len(self.subject_codes), text_dim)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device).eval()

    def test_dataset(self, sent2emb: Dict[str, Sequence[float]] = None) -> StreamingEEGDataset:
        """The held-out test trials, in the same order as the full dataset."""
        dataset = StreamingEEGDataset(
            root=self.chisco_root,
            task=self.task,
            subject_codes=self.subject_codes,
            sentences=self.split["test_sentences"],
            sent2emb=sent2emb,
        )
        if len(dataset) == 0:
            raise ValueError(f"No test trials found for {self.run_dir}")
        return dataset


@torch.no_grad()
def encode_eeg(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, List[str]]:
    """Embed every trial of ``dataset``. Returns L2-normalised embeddings (N, D) and the N sentences."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings, sentences = [], []
    for batch in loader:
        emb = model(batch["eeg"].to(device), batch["subject_id"].to(device))
        embeddings.append(nn.functional.normalize(emb, dim=1))
        sentences.extend(batch["sentence"])
    return torch.cat(embeddings), sentences


def top_k_retrieval(
    eeg_emb: torch.Tensor,
    sentences: Sequence[str],
    text_matrix: torch.Tensor,
    sentence_to_id: Dict[str, int],
    ks: Sequence[int] = (1, 5, 10, 50),
) -> Tuple[Dict[str, float], List[List[int]], List[int]]:
    """Rank every candidate sentence for each trial by cosine similarity.

    Returns:
        accuracy:   {"top1": %, "top5": %, ...} over all trials
        candidates: for each trial, the ids of its top max(ks) candidates
        true_ids:   for each trial, the id of the correct sentence (None if it has no embedding)
    """
    k_max = min(max(ks), text_matrix.shape[0])
    sims = eeg_emb @ text_matrix.to(eeg_emb.device).T
    candidates = sims.topk(k=k_max, dim=1).indices.tolist()
    true_ids = [sentence_to_id.get(s) for s in sentences]

    n = len(sentences)
    accuracy = {
        f"top{k}": 100.0 * sum(t is not None and t in c[:k] for t, c in zip(true_ids, candidates)) / n
        for k in ks
    }
    return accuracy, candidates, true_ids

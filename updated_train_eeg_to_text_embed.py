# train_eeg_to_text_embed_cli.py
#
# CLI version of train_eeg_to_text_embed.py:
# - argparse for ALL config/hyperparams
# - toggles for model parts: EEGNet frontend, spatial attention, subject layer, residual blocks
# - experiment logging via SimpleExperimentLogger (config.json, split.json, metrics.jsonl, best/last checkpoints)
# - plots saved inside experiment run folder
#
# Based on the user's current training script. :contentReference[oaicite:2]{index=2}

import os
import sys
import math
import json
import random
import argparse
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Ensure we can import ds005170 modules
sys.path.append("./ds005170")

from simple_experiment_logger import SimpleExperimentLogger
from Brain_Module import BrainModule
from CHISCO_baseline_model import CHISCOBaselineEEGEncoder
from data_loader import (
    load_chisco_imagine_all_subjects,
    load_chisco_read_all_subjects,
    load_chisco_combined_all_subjects,
    ChiscoImagineDataset,
)
from model_parts import EEGFrontEnd


# ----------------- Utils -----------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def uppercase_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build a JSON-serializable config dict for logging.
    """
    d = vars(args).copy()
    # Make sure everything is JSON-friendly
    for k, v in list(d.items()):
        if isinstance(v, torch.device):
            d[k] = str(v)
    return d


# ----------------- Data prep -----------------

def subsample_train_indices_global(train_idx, fraction, seed):
    if fraction >= 1.0:
        return train_idx
    rng = random.Random(seed)
    idx = list(train_idx)
    rng.shuffle(idx)
    k = max(1, int(len(idx) * fraction))
    return idx[:k]


def load_sentence_embeddings(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eeg_text_embed_tensors(
    chisco_root: str,
    sent_emb_path: str,
    task: str,
    subject_codes: List[str],
) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
    """
    Returns:
        X_kept:        (N, C, T) float32 tensor
        sentences_kept:list[str] length N
        subj_ids_kept: (N,) long tensor
        text_targets:  (N, D) float32 tensor
    """
    print(f"Loading CHISCO {task} EEG + sentences...")
    if task == "read":
        X, sentences, subj_ids = load_chisco_read_all_subjects(root=chisco_root, subject_codes=subject_codes)
    elif task == "imagine":
        X, sentences, subj_ids = load_chisco_imagine_all_subjects(root=chisco_root, subject_codes=subject_codes)
    elif task == "combined":
        X, sentences, subj_ids = load_chisco_combined_all_subjects(root=chisco_root, subject_codes=subject_codes)
    else:
        raise ValueError("task must be 'read' or 'imagine'")

    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints.")

    print("Loading precomputed sentence embeddings...")
    sent2emb = load_sentence_embeddings(sent_emb_path)

    keep_indices: List[int] = []
    targets: List[torch.Tensor] = []
    missing = 0

    for i, sent in enumerate(sentences):
        emb = sent2emb.get(sent)
        if emb is None:
            missing += 1
            continue
        keep_indices.append(i)
        targets.append(torch.tensor(emb, dtype=torch.float32))

    if missing:
        print(f"Warning: {missing} sentences missing in SENT_EMB_PATH; skipped.")

    X_kept = X[keep_indices]
    subj_ids_kept = subj_ids[keep_indices]
    sentences_kept = [sentences[i] for i in keep_indices]
    text_targets = torch.stack(targets, dim=0)

    print(f"Using {len(keep_indices)} trials with valid text embeddings.")
    print(f"text_dim = {text_targets.shape[1]}")
    return X_kept, sentences_kept, subj_ids_kept, text_targets


# ----------------- Splits -----------------

def make_train_val_indices(n_samples: int, val_split: float, shuffle: bool, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n_samples))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
    val_size = int(n_samples * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return train_idx, val_idx


def make_split_indices(
    split_mode: str,
    sentences: List[str],
    subj_ids: torch.Tensor,
    subject_codes: List[str],
    seed: int,
    val_split: float,
    shuffle: bool,
    holdout_subjects: List[str],
    holdout_sent_fraction: float,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Returns: train_idx, val_idx, split_info

    Functionality is identical to the original version; the only change is that
    we also store the explicit train/val indices inside split_info for exact reproducibility.
    """
    n = len(sentences)

    if split_mode == "iid":
        train_idx, val_idx = make_train_val_indices(n, val_split, shuffle, seed)
        split_info = {
            "split_mode": "iid",
            "val_split": float(val_split),
            "seed": int(seed),
            "shuffle": bool(shuffle),
            # NEW:
            "train_idx": [int(i) for i in train_idx],
            "val_idx": [int(i) for i in val_idx],
        }
        print(f"Train size: {len(train_idx)}, val size: {len(val_idx)}")
        return train_idx, val_idx, split_info

    if split_mode == "unseen_subject":
        code_to_idx = {code: i for i, code in enumerate(subject_codes)}
        holdout_ids = {code_to_idx[c] for c in holdout_subjects}

        # NOTE: subj_ids values in your pipeline behave like integer indices already
        val_idx = [i for i in range(n) if int(subj_ids[i]) in holdout_ids]
        train_idx = [i for i in range(n) if int(subj_ids[i]) not in holdout_ids]

        split_info = {
            "split_mode": "unseen_subject",
            "holdout_subjects": list(holdout_subjects),
            "holdout_subject_indices": sorted(list(holdout_ids)),
            "seed": int(seed),
            # NEW:
            "train_idx": [int(i) for i in train_idx],
            "val_idx": [int(i) for i in val_idx],
        }
        print(f"Held out subjects {holdout_subjects} -> val trials: {len(val_idx)}")
        print(f"Train size: {len(train_idx)}, val size: {len(val_idx)}")
        return train_idx, val_idx, split_info

    if split_mode == "unseen_sentence":
        rng = random.Random(seed)
        unique_sents = sorted(set(sentences))
        rng.shuffle(unique_sents)

        n_holdout = max(1, int(len(unique_sents) * holdout_sent_fraction))
        holdout_sents = set(unique_sents[:n_holdout])

        val_idx = [i for i, s in enumerate(sentences) if s in holdout_sents]
        train_idx = [i for i, s in enumerate(sentences) if s not in holdout_sents]

        split_info = {
            "split_mode": "unseen_sentence",
            "holdout_fraction": float(holdout_sent_fraction),
            "holdout_count": int(n_holdout),
            "seed": int(seed),
            # NEW:
            "train_idx": [int(i) for i in train_idx],
            "val_idx": [int(i) for i in val_idx],
            # Optional (doesn't change behavior, but nice for debugging):
            # "holdout_sentences": sorted(list(holdout_sents)),
        }
        print(f"Held out {n_holdout}/{len(unique_sents)} sentences -> val trials: {len(val_idx)}")
        print(f"Train size: {len(train_idx)}, val size: {len(val_idx)}")
        return train_idx, val_idx, split_info

    raise ValueError(f"Unknown split_mode: {split_mode}")


# ----------------- Loss & metrics -----------------

def contrastive_loss(eeg_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float):
    eeg_emb = nn.functional.normalize(eeg_emb, dim=-1)
    text_emb = nn.functional.normalize(text_emb, dim=-1)

    logits = eeg_emb @ text_emb.T / temperature
    labels = torch.arange(eeg_emb.size(0), device=eeg_emb.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2.0, logits


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, temperature: float):
    model.eval()
    total_loss = 0.0
    total = 0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_acc10 = 0.0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        text = batch["target"].to(device)

        eeg_emb = model(eeg, subj)
        loss, logits = contrastive_loss(eeg_emb, text, temperature)

        B = eeg.size(0)
        total_loss += loss.item() * B
        total += B

        targets = torch.arange(B, device=device)
        _, topk = logits.topk(k=min(10, B), dim=1)
        correct = topk.eq(targets.unsqueeze(1))

        acc1 = correct[:, :1].any(dim=1).float().mean().item() * 100.0
        acc5 = correct[:, :5].any(dim=1).float().mean().item() * 100.0 if B >= 5 else acc1
        acc10 = correct[:, :10].any(dim=1).float().mean().item() * 100.0 if B >= 10 else acc5

        total_acc1 += acc1 * B
        total_acc5 += acc5 * B
        total_acc10 += acc10 * B

    return (
        total_loss / total,
        total_acc1 / total,
        total_acc5 / total,
        total_acc10 / total,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    temperature: float,
    log_every_steps: int,
):
    model.train()
    total_loss = 0.0
    total = 0

    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        text = batch["target"].to(device)

        optimizer.zero_grad()
        eeg_emb = model(eeg, subj)
        loss, _ = contrastive_loss(eeg_emb, text, temperature)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * eeg.size(0)
        total += eeg.size(0)

        if log_every_steps > 0 and (step + 1) % log_every_steps == 0:
            print(f"Epoch {epoch} | Step {step+1}/{len(loader)} | Loss {(total_loss/total):.4f}")

    return total_loss / total


# ----------------- Model -----------------

class EEGToTextEmbedModel(nn.Module):
    """
    Optional frontend -> optional BrainModule -> always-on prediction head.
    """
    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        text_dim: int,
        latent_dim: int,
        hidden_channels: int,
        final_hidden: int,
        n_blocks: int,
        pooling: str,
        # stage toggles
        use_frontend: bool,
        use_brainmodule: bool,
        # frontend toggles
        use_eegnet: bool,
        eegnet_temporal_kern: int,
        eegnet_F1: int,
        eegnet_D: int,
        raw_frontend_out_ch: int,
        # brainmodule toggles
        use_spatial_attn: bool,
        use_subject_layer: bool,
        use_residual_blocks: bool,
    ):
        super().__init__()
        self.pooling = pooling
        self.use_frontend = use_frontend
        self.use_brainmodule = use_brainmodule

        # 1) Frontend (optional)
        if self.use_frontend:
            if use_eegnet:
                self.eeg_frontend = EEGFrontEnd(
                    in_ch=n_channels,
                    temporal_kern=eegnet_temporal_kern,
                    F1=eegnet_F1,
                    D=eegnet_D,
                )
                frontend_out_ch = eegnet_F1 * eegnet_D
            else:
                # Simple alternative: 1x1 conv to map raw channels to a chosen feature width
                self.eeg_frontend = nn.Conv1d(n_channels, raw_frontend_out_ch, kernel_size=1)
                frontend_out_ch = raw_frontend_out_ch
        else:
            self.eeg_frontend = nn.Identity()
            frontend_out_ch = n_channels

        # 2) BrainModule (optional)
        if self.use_brainmodule:
            self.brain = BrainModule(
                in_channels=frontend_out_ch,
                out_channels=latent_dim,
                n_subjects=n_subjects,
                hidden_channels=hidden_channels,
                final_hidden=final_hidden,
                n_blocks=n_blocks,
                spatial_out_channels=frontend_out_ch,
                use_spatial_attn=use_spatial_attn,
                use_subject_layer=use_subject_layer,
                use_residual_blocks=use_residual_blocks,
            )
            self.sensor_pos = nn.Parameter(torch.randn(frontend_out_ch, 2) * 0.1)
            pred_in_dim = latent_dim
        else:
            self.brain = None
            self.sensor_pos = None
            pred_in_dim = frontend_out_ch

        # 3) Prediction head (always on)
        self.proj = nn.Sequential(
            nn.Linear(pred_in_dim, pred_in_dim),
            nn.ReLU(),
            nn.Linear(pred_in_dim, text_dim),
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        x = self.eeg_frontend(x)  # (B, C', T')

        if self.brain is not None:
            z = self.brain(x, self.sensor_pos, subject_ids)  # (B, latent_dim, T')
        else:
            z = x  # bypass BrainModule entirely

        if self.pooling == "mean":
            z = z.mean(dim=-1)
        elif self.pooling == "max":
            z, _ = z.max(dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        e = self.proj(z)
        return nn.functional.normalize(e, dim=-1)

class _CHISCOBaselineWrapper(nn.Module):
    """
    Keeps the same call signature as your current model:
      forward(eeg, subject_ids) -> embedding
    even though CHISCO baseline doesn't use subject_ids.
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ----------------- CLI -----------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train EEG→text embedding model (CHISCO) with contrastive loss (CLI)."
    )

    # Paths / data
    p.add_argument("--chisco-root", type=str, default="./ds005170/derivatives/preprocessed_pkl")
    p.add_argument("--sent-emb-path", type=str, default="./ds005170/json/sentence_bert_embs.json")
    p.add_argument("--task", type=str, default="imagine", choices=["read", "imagine", "combined"])
    p.add_argument("--subject-codes", type=str, default="01,02,03,04,05",
                   help="Comma-separated subject codes, e.g. 01,02,03,04,05")
    p.add_argument("--train-data-fraction", type=float, default=1.0)

    # Split
    p.add_argument("--split-mode", type=str, default="unseen_sentence",
                   choices=["iid", "unseen_subject", "unseen_sentence"])
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--shuffle-data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-subjects", type=str, default="05",
                   help="Comma-separated subject codes to hold out, e.g. 05")
    p.add_argument("--holdout-sent-fraction", type=float, default=0.2)

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.0002)
    p.add_argument("--weight-decay", type=float, default=1e-05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every-steps", type=int, default=50)

    # Dataset augmentation
    p.add_argument("--augment-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--jitter-max", type=int, default=10)
    p.add_argument("--channel-dropout-p", type=float, default=0.05)

    # Model dims
    p.add_argument("--hidden-channels", type=int, default=256)
    p.add_argument("--final-hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])

    # Toggle model parts
    p.add_argument("--use-frontend", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable stage 1 frontend. If false, raw EEG goes directly to the BrainModule or prediction head.")
    p.add_argument("--use-brainmodule", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable stage 2 BrainModule. If false, frontend/raw features go directly to the prediction head.")
    p.add_argument("--use-eegnet", action=argparse.BooleanOptionalAction, default=True,
                   help="If stage 1 frontend is enabled, use EEGNet-style frontend. If false, uses a simple 1x1 conv.")
    p.add_argument("--eegnet-temporal-kern", type=int, default=64)
    p.add_argument("--eegnet-f1", type=int, default=16)
    p.add_argument("--eegnet-d", type=int, default=2)
    p.add_argument("--raw-frontend-out-ch", type=int, default=32,
                   help="If --no-use-eegnet: output channels of the simple 1x1 conv frontend.")

    p.add_argument("--use-spatial-attn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-subject-layer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-residual-blocks", action=argparse.BooleanOptionalAction, default=True)

    # Baseline stuff
    p.add_argument("--arch", type=str, default="brainmodule",
                   choices=["brainmodule", "chisco_baseline"])

    p.add_argument("--baseline-layer", type=int, default=0)  # transformer layers (like CHISCO)
    p.add_argument("--baseline-pooling", type=str, default="mean",
                   choices=["mean", "sums", "tops"])

    p.add_argument("--baseline-size1", type=int, default=8)
    p.add_argument("--baseline-size2", type=int, default=8)
    p.add_argument("--baseline-feel1", type=int, default=125)
    p.add_argument("--baseline-feel2", type=int, default=25)
    p.add_argument("--baseline-dropout1", type=float, default=0.1)
    p.add_argument("--baseline-dropout2", type=float, default=0.1)
    p.add_argument("--baseline-latent-dim", type=int, default=256)

    # Logging / output
    p.add_argument("--exp-root", type=str, default="experiments")
    p.add_argument("--experiment-name", type=str, default="eeg2text")
    p.add_argument("--no-plots", action=argparse.BooleanOptionalAction, default=False)

    # Device
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return p


def main():
    args = build_argparser().parse_args()

    subject_codes = [s.strip() for s in args.subject_codes.split(",") if s.strip()]
    holdout_subjects = [s.strip() for s in args.holdout_subjects.split(",") if s.strip()]

    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # Load data
    X, sentences, subj_ids, text_targets = build_eeg_text_embed_tensors(
        chisco_root=args.chisco_root,
        sent_emb_path=args.sent_emb_path,
        task=args.task,
        subject_codes=subject_codes,
    )

    n_samples, n_channels, _ = X.shape
    text_dim = text_targets.shape[1]

    # Split
    train_idx, val_idx, split_info = make_split_indices(
        split_mode=args.split_mode,
        sentences=sentences,
        subj_ids=subj_ids,
        subject_codes=subject_codes,
        seed=args.seed,
        val_split=args.val_split,
        shuffle=args.shuffle_data,
        holdout_subjects=holdout_subjects,
        holdout_sent_fraction=args.holdout_sent_fraction,
    )

    train_idx = subsample_train_indices_global(
        train_idx=train_idx,
        fraction=args.train_data_fraction,
        seed=args.seed,
    )

    split_info["train_data_fraction"] = float(args.train_data_fraction)
    split_info["train_size_after_fraction"] = len(train_idx)

    # Logger path structure: <exp-root>/<experiment-name>/<task>/<split-mode>/<run_id>/
    task_name = os.path.join(args.experiment_name, args.task, split_info["split_mode"])
    config_dict = uppercase_config_from_args(args)
    config_dict["subject_codes"] = subject_codes
    config_dict["holdout_subjects"] = holdout_subjects

    logger = SimpleExperimentLogger(
        root=args.exp_root,
        task_name=task_name,
        config=config_dict,
        split_info=split_info,
        metric_name="val_acc10",
        higher_is_better=True,
        run_tag=None,
    )
    print("Logging to:", logger.run_dir)

    # Datasets
    train_dataset = ChiscoImagineDataset(
        eeg=X[train_idx],
        sentences=[sentences[i] for i in train_idx],
        subject_ids=subj_ids[train_idx],
        targets=text_targets[train_idx],
        augment=args.augment_train,
        noise_std=args.noise_std,
        jitter_max=args.jitter_max,
        channel_dropout_p=args.channel_dropout_p,
    )

    val_dataset = ChiscoImagineDataset(
        eeg=X[val_idx],
        sentences=[sentences[i] for i in val_idx],
        subject_ids=subj_ids[val_idx],
        targets=text_targets[val_idx],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    n_subjects = len(subject_codes)

    if args.arch == "brainmodule":
        if (not args.use_brainmodule) and (
            args.use_spatial_attn or args.use_subject_layer or args.use_residual_blocks
        ):
            print("Warning: BrainModule is disabled, so --use-spatial-attn/--use-subject-layer/--use-residual-blocks are ignored.")

        model = EEGToTextEmbedModel(
            n_channels=n_channels,
            n_subjects=n_subjects,
            text_dim=text_dim,
            latent_dim=args.latent_dim,
            hidden_channels=args.hidden_channels,
            final_hidden=args.final_hidden,
            n_blocks=args.n_blocks,
            pooling=args.pooling,
            use_frontend=args.use_frontend,
            use_brainmodule=args.use_brainmodule,
            use_eegnet=args.use_eegnet,
            eegnet_temporal_kern=args.eegnet_temporal_kern,
            eegnet_F1=args.eegnet_f1,
            eegnet_D=args.eegnet_d,
            raw_frontend_out_ch=args.raw_frontend_out_ch,
            use_spatial_attn=args.use_spatial_attn,
            use_subject_layer=args.use_subject_layer,
            use_residual_blocks=args.use_residual_blocks,
        ).to(device)

    elif args.arch == "chisco_baseline":
        # Baseline ignores subject ids (kept in the training loop unchanged)
        model = _CHISCOBaselineWrapper(
            encoder=CHISCOBaselineEEGEncoder(
                n_channels=n_channels,
                text_dim=text_dim,
                latent_dim=args.baseline_latent_dim,
                dropout1=args.baseline_dropout1,
                dropout2=args.baseline_dropout2,
                feel1=args.baseline_feel1,
                feel2=args.baseline_feel2,
                size1=args.baseline_size1,
                size2=args.baseline_size2,
                layer=args.baseline_layer,
                pooling=args.baseline_pooling,
            )
        ).to(device)

    else:
        raise ValueError(args.arch)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Train
    best_val_acc10 = -1e9
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_acc1_hist: List[float] = []
    val_acc5_hist: List[float] = []
    val_acc10_hist: List[float] = []

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.num_epochs} ===")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            temperature=args.temperature,
            log_every_steps=args.log_every_steps,
        )
        print(f"Train   | Loss {train_loss:.4f}")

        val_loss, val_acc1, val_acc5, val_acc10 = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            temperature=args.temperature,
        )
        print(
            f"Val     | Loss {val_loss:.4f} | Acc@1 {val_acc1:.2f}% "
            f"| Acc@5 {val_acc5:.2f}% | Acc@10 {val_acc10:.2f}%"
        )

        # logger: metrics
        logger.log_epoch(
            epoch=epoch,
            train_stats={"loss": float(train_loss)},
            val_stats={"loss": float(val_loss), "acc1": float(val_acc1), "acc5": float(val_acc5), "acc10": float(val_acc10)},
        )

        # logger: last checkpoint
        logger.save_last({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config_dict,
            "split_info": split_info,
            "train": {"loss": float(train_loss)},
            "val": {"loss": float(val_loss), "acc1": float(val_acc1), "acc5": float(val_acc5), "acc10": float(val_acc10)},
        })

        # logger: best checkpoint
        if val_acc10 > best_val_acc10:
            best_val_acc10 = val_acc10
            did_save = logger.maybe_save_best(
                epoch=epoch,
                metric_value=float(val_acc10),
                checkpoint={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config_dict,
                    "split_info": split_info,
                    "best_val_acc10": float(best_val_acc10),
                    "train": {"loss": float(train_loss)},
                    "val": {"loss": float(val_loss), "acc1": float(val_acc1), "acc5": float(val_acc5), "acc10": float(val_acc10)},
                },
            )
            if did_save:
                print(f"Saved NEW BEST -> {logger.run_dir}/best_checkpoint.pt (val_acc@10={best_val_acc10:.2f}%)")

        train_loss_hist.append(float(train_loss))
        val_loss_hist.append(float(val_loss))
        val_acc1_hist.append(float(val_acc1))
        val_acc5_hist.append(float(val_acc5))
        val_acc10_hist.append(float(val_acc10))

    # Plots inside experiment folder
    if not args.no_plots:
        plots_dir = os.path.join(logger.run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        epochs = range(1, args.num_epochs + 1)

        plt.figure()
        plt.plot(epochs, train_loss_hist, label="Train Loss")
        plt.plot(epochs, val_loss_hist, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("EEG→Text contrastive loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "eeg2text_loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, val_acc1_hist, label="Val Acc@1")
        plt.plot(epochs, val_acc5_hist, label="Val Acc@5")
        plt.plot(epochs, val_acc10_hist, label="Val Acc@10")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("EEG→Text retrieval (within-batch)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "eeg2text_acc_curve.png"))
        plt.close()

        print("Saved plots to:", plots_dir)

    print(f"\nTraining finished. Best val Acc@10: {best_val_acc10:.2f}%")
    print("Run directory:", logger.run_dir)


if __name__ == "__main__":
    main()

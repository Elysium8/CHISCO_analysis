# train_eeg_to_concreteness_cli.py
#
# Concreteness regression version of updated_train_eeg_to_text_embed.py
# - CLI args for all settings (same style)
# - Same split logic and logging style
# - Same model backbone (EEGFrontEnd + BrainModule)
# - Different target: sentence-level concreteness from JSON mapping
# - Different loss: MSELoss (regression)
#
# IMPORTANT: concreteness JSON format must be: { "<sentence>": <float>, ... }
# where float is ideally in [0, 1].

import os
import sys
import math
import json
import random
import argparse
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Ensure we can import ds005170 modules
sys.path.append("./ds005170")

from simple_experiment_logger import SimpleExperimentLogger
from Brain_Module import BrainModule
from data_loader import (
    load_chisco_imagine_all_subjects,
    load_chisco_read_all_subjects,
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
    for k, v in list(d.items()):
        if isinstance(v, torch.device):
            d[k] = str(v)
    return d


# ----------------- Data prep -----------------

def load_concreteness_map(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # coerce to float
    return {str(k): float(v) for k, v in data.items()}


def build_eeg_concreteness_tensors(
    chisco_root: str,
    concreteness_json: str,
    task: str,
    subject_codes: List[str],
) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
    """
    Returns:
        X_kept:        (N, C, T) float32 tensor
        sentences_kept:list[str] length N
        subj_ids_kept: (N,) long tensor
        y:             (N,) float32 tensor (concreteness)
    """
    print(f"Loading CHISCO {task} EEG + sentences...")
    if task == "read":
        X, sentences, subj_ids = load_chisco_read_all_subjects(root=chisco_root, subject_codes=subject_codes)
    elif task == "imagine":
        X, sentences, subj_ids = load_chisco_imagine_all_subjects(root=chisco_root, subject_codes=subject_codes)
    else:
        raise ValueError("task must be 'read' or 'imagine'")

    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints.")

    print("Loading sentence concreteness mapping...")
    sent2conc = load_concreteness_map(concreteness_json)

    keep_indices: List[int] = []
    targets: List[float] = []
    missing = 0

    # IMPORTANT: mapping uses exact sentence strings as keys.
    for i, sent in enumerate(sentences):
        if sent not in sent2conc:
            missing += 1
            continue
        keep_indices.append(i)
        targets.append(float(sent2conc[sent]))

    if missing:
        print(f"Warning: {missing} sentences missing in CONCRETENESS_JSON; skipped.")

    if len(keep_indices) == 0:
        raise RuntimeError(
            "No sentences matched the concreteness JSON keys. "
            "Check that your JSON uses EXACT same sentence strings."
        )

    X_kept = X[keep_indices]
    subj_ids_kept = subj_ids[keep_indices]
    sentences_kept = [sentences[i] for i in keep_indices]
    y = torch.tensor(targets, dtype=torch.float32)

    print(f"Using {len(keep_indices)} trials with valid concreteness targets.")
    print(f"Concreteness target stats: min={y.min().item():.3f}, max={y.max().item():.3f}, mean={y.mean().item():.3f}")
    return X_kept, sentences_kept, subj_ids_kept, y


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
    Identical split behavior to your original, plus stores explicit indices for reproducibility.
    """
    n = len(sentences)

    if split_mode == "iid":
        train_idx, val_idx = make_train_val_indices(n, val_split, shuffle, seed)
        split_info = {
            "split_mode": "iid",
            "val_split": float(val_split),
            "seed": int(seed),
            "shuffle": bool(shuffle),
            "train_idx": [int(i) for i in train_idx],
            "val_idx": [int(i) for i in val_idx],
        }
        print(f"Train size: {len(train_idx)}, val size: {len(val_idx)}")
        return train_idx, val_idx, split_info

    if split_mode == "unseen_subject":
        code_to_idx = {code: i for i, code in enumerate(subject_codes)}
        holdout_ids = {code_to_idx[c] for c in holdout_subjects}

        val_idx = [i for i in range(n) if int(subj_ids[i]) in holdout_ids]
        train_idx = [i for i in range(n) if int(subj_ids[i]) not in holdout_ids]

        split_info = {
            "split_mode": "unseen_subject",
            "holdout_subjects": list(holdout_subjects),
            "holdout_subject_indices": sorted(list(holdout_ids)),
            "seed": int(seed),
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
            "train_idx": [int(i) for i in train_idx],
            "val_idx": [int(i) for i in val_idx],
        }
        print(f"Held out {n_holdout}/{len(unique_sents)} sentences -> val trials: {len(val_idx)}")
        print(f"Train size: {len(train_idx)}, val size: {len(val_idx)}")
        return train_idx, val_idx, split_info

    raise ValueError(f"Unknown split_mode: {split_mode}")


# ----------------- Loss & metrics -----------------

def pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    # x,y: (N,)
    x = x.detach().float()
    y = y.detach().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).item()
    if denom < 1e-12:
        return 0.0
    return float((x @ y).item() / denom)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total = 0

    all_y = []
    all_yhat = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        y = batch["target"].to(device).float()

        yhat = model(eeg, subj)
        loss = criterion(yhat, y)

        B = eeg.size(0)
        total_loss += loss.item() * B
        total += B

        all_y.append(y.detach())
        all_yhat.append(yhat.detach())

    y = torch.cat(all_y, dim=0)
    yhat = torch.cat(all_yhat, dim=0)

    mae = torch.mean(torch.abs(yhat - y)).item()
    pr = pearsonr_torch(yhat, y)

    return (total_loss / max(1, total)), mae, pr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    criterion: nn.Module,
    log_every_steps: int,
):
    model.train()
    total_loss = 0.0
    total = 0

    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        y = batch["target"].to(device).float()

        optimizer.zero_grad()
        yhat = model(eeg, subj)
        loss = criterion(yhat, y)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * eeg.size(0)
        total += eeg.size(0)

        if log_every_steps > 0 and (step + 1) % log_every_steps == 0:
            print(f"Epoch {epoch} | Step {step+1}/{len(loader)} | Loss {(total_loss/total):.4f}")

    return total_loss / max(1, total)


# ----------------- Model -----------------

class EEGToConcretenessModel(nn.Module):
    """
    EEG frontend (optional) -> BrainModule -> pooling -> scalar head
    (same backbone as EEGToTextEmbedModel, just different head)
    """
    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        latent_dim: int,
        hidden_channels: int,
        final_hidden: int,
        n_blocks: int,
        pooling: str,
        # toggles
        use_eegnet: bool,
        eegnet_temporal_kern: int,
        eegnet_F1: int,
        eegnet_D: int,
        # brainmodule toggles
        use_spatial_attn: bool,
        use_subject_layer: bool,
        use_residual_blocks: bool,
        # simple fallback frontend if not using EEGNet
        raw_frontend_out_ch: int,
    ):
        super().__init__()
        self.pooling = pooling

        if use_eegnet:
            self.eeg_frontend = EEGFrontEnd(
                in_ch=n_channels,
                temporal_kern=eegnet_temporal_kern,
                F1=eegnet_F1,
                D=eegnet_D,
            )
            frontend_out_ch = eegnet_F1 * eegnet_D
        else:
            self.eeg_frontend = nn.Conv1d(n_channels, raw_frontend_out_ch, kernel_size=1)
            frontend_out_ch = raw_frontend_out_ch

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

        # scalar regression head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        x = self.eeg_frontend(x)  # (B, C', T')
        z = self.brain(x, self.sensor_pos, subject_ids)  # (B, latent_dim, T')

        if self.pooling == "mean":
            z = z.mean(dim=-1)
        elif self.pooling == "max":
            z, _ = z.max(dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        yhat = self.head(z).squeeze(-1)  # (B,)
        return yhat


# ----------------- CLI -----------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train EEG→concreteness model (CHISCO) (CLI)."
    )

    # Paths / data
    p.add_argument("--chisco-root", type=str, default="./ds005170/derivatives/preprocessed_pkl")
    p.add_argument("--concreteness-json", type=str, default="./ds005170/json/chisco_concreteness_map_heuristic.json",
                   help="JSON mapping: {sentence: float}")
    p.add_argument("--task", type=str, default="imagine", choices=["read", "imagine"])
    p.add_argument("--subject-codes", type=str, default="01,02,03,04,05",
                   help="Comma-separated subject codes, e.g. 01,02,03,04,05")

    # Split
    p.add_argument("--split-mode", type=str, default="iid",
                   choices=["iid", "unseen_subject", "unseen_sentence"])
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--shuffle-data", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-subjects", type=str, default="05",
                   help="Comma-separated subject codes to hold out, e.g. 05")
    p.add_argument("--holdout-sent-fraction", type=float, default=0.2)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
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
    p.add_argument("--use-eegnet", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eegnet-temporal-kern", type=int, default=64)
    p.add_argument("--eegnet-f1", type=int, default=16)
    p.add_argument("--eegnet-d", type=int, default=2)
    p.add_argument("--raw-frontend-out-ch", type=int, default=32)

    p.add_argument("--use-spatial-attn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-subject-layer", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-residual-blocks", action=argparse.BooleanOptionalAction, default=True)

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
    X, sentences, subj_ids, y = build_eeg_concreteness_tensors(
        chisco_root=args.chisco_root,
        concreteness_json=args.concreteness_json,
        task=args.task,
        subject_codes=subject_codes,
    )

    n_samples, n_channels, _ = X.shape

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

    # Logger path structure: <exp-root>/<experiment-name>/<task>/<split-mode>/<run_id>/
    task_name = os.path.join(args.experiment_name, f"{args.task}_concreteness", split_info["split_mode"])
    config_dict = uppercase_config_from_args(args)
    config_dict["subject_codes"] = subject_codes
    config_dict["holdout_subjects"] = holdout_subjects

    logger = SimpleExperimentLogger(
        root=args.exp_root,
        task_name=task_name,
        config=config_dict,
        split_info=split_info,
        metric_name="val_pearson",
        higher_is_better=True,
        run_tag=f"seed={args.seed}",
    )
    print("Logging to:", logger.run_dir)

    # Datasets (targets are scalar floats)
    train_dataset = ChiscoImagineDataset(
        eeg=X[train_idx],
        sentences=[sentences[i] for i in train_idx],
        subject_ids=subj_ids[train_idx],
        targets=y[train_idx],
        augment=args.augment_train,
        noise_std=args.noise_std,
        jitter_max=args.jitter_max,
        channel_dropout_p=args.channel_dropout_p,
    )

    val_dataset = ChiscoImagineDataset(
        eeg=X[val_idx],
        sentences=[sentences[i] for i in val_idx],
        subject_ids=subj_ids[val_idx],
        targets=y[val_idx],
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
    model = EEGToConcretenessModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        latent_dim=args.latent_dim,
        hidden_channels=args.hidden_channels,
        final_hidden=args.final_hidden,
        n_blocks=args.n_blocks,
        pooling=args.pooling,
        use_eegnet=args.use_eegnet,
        eegnet_temporal_kern=args.eegnet_temporal_kern,
        eegnet_F1=args.eegnet_f1,
        eegnet_D=args.eegnet_d,
        use_spatial_attn=args.use_spatial_attn,
        use_subject_layer=args.use_subject_layer,
        use_residual_blocks=args.use_residual_blocks,
        raw_frontend_out_ch=args.raw_frontend_out_ch,
    ).to(device)

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

    criterion = nn.MSELoss()

    # Train
    best_val_pearson = -1e9
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_mae_hist: List[float] = []
    val_pearson_hist: List[float] = []

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.num_epochs} ===")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            criterion=criterion,
            log_every_steps=args.log_every_steps,
        )
        print(f"Train   | Loss {train_loss:.4f}")

        val_loss, val_mae, val_pearson = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
        )
        print(f"Val     | Loss {val_loss:.4f} | MAE {val_mae:.4f} | Pearson r {val_pearson:.4f}")

        logger.log_epoch(
            epoch=epoch,
            train_stats={"loss": float(train_loss)},
            val_stats={"loss": float(val_loss), "mae": float(val_mae), "pearson": float(val_pearson)},
        )

        logger.save_last({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config_dict,
            "split_info": split_info,
            "train": {"loss": float(train_loss)},
            "val": {"loss": float(val_loss), "mae": float(val_mae), "pearson": float(val_pearson)},
        })

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            did_save = logger.maybe_save_best(
                epoch=epoch,
                metric_value=float(val_pearson),
                checkpoint={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config_dict,
                    "split_info": split_info,
                    "best_val_pearson": float(best_val_pearson),
                    "train": {"loss": float(train_loss)},
                    "val": {"loss": float(val_loss), "mae": float(val_mae), "pearson": float(val_pearson)},
                },
            )
            if did_save:
                print(f"Saved NEW BEST -> {logger.run_dir}/best_checkpoint.pt (val_pearson={best_val_pearson:.4f})")

        train_loss_hist.append(float(train_loss))
        val_loss_hist.append(float(val_loss))
        val_mae_hist.append(float(val_mae))
        val_pearson_hist.append(float(val_pearson))

    # Plots inside experiment folder
    if not args.no_plots:
        plots_dir = os.path.join(logger.run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        epochs = range(1, args.num_epochs + 1)

        plt.figure()
        plt.plot(epochs, train_loss_hist, label="Train Loss")
        plt.plot(epochs, val_loss_hist, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("EEG→Concreteness regression loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "concreteness_loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, val_mae_hist, label="Val MAE")
        plt.plot(epochs, val_pearson_hist, label="Val Pearson r")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("EEG→Concreteness regression metrics")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "concreteness_metrics_curve.png"))
        plt.close()

        print("Saved plots to:", plots_dir)

    print(f"\nTraining finished. Best val Pearson r: {best_val_pearson:.4f}")
    print("Run directory:", logger.run_dir)


if __name__ == "__main__":
    main()

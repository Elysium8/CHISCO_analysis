# train_sentence_length.py
#
# Train EEG -> sentence length regressor on CHISCO (ds005170).
# Uses:
#   - build_length_dataset.build_sentence_length_tensors
#   - models_regression.EEGToSentenceLengthModel
#   - data_loader.ChiscoImagineDataset (generic dataset wrapper + augmentations)
#
# Metrics printed per epoch:
#   - Loss (SmoothL1 in log space)
#   - MAE / RMSE in chars
#   - RMSE in log1p(chars)
#   - R^2 in chars

import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Ensure we can import ds005170 modules if running from repo root
import sys
sys.path.append("./ds005170")

from build_length_dataset import build_sentence_length_tensors
from models_regression import EEGToSentenceLengthModel
from data_loader import ChiscoImagineDataset


# ----------------- Config -----------------

@dataclass
class Config:
    # Paths
    CHISCO_ROOT: str = "./ds005170/derivatives/preprocessed_pkl"
    SENT_LEN_PATH: str = "./ds005170/json/sentence_length_chars.json"

    # Subjects
    SUBJECT_CODES: List[str] = None

    # Training
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 40
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    VAL_SPLIT: float = 0.2
    SHUFFLE_DATA: bool = True
    RANDOM_SEED: int = 42
    WARMUP_RATIO: float = 0.1

    # Model
    HIDDEN_CHANNELS: int = 256
    FINAL_HIDDEN: int = 256
    N_BLOCKS: int = 5
    LATENT_DIM: int = 256
    ATTN_HIDDEN: int = 128
    USE_SUBJ_EMB_IN_HEAD: bool = True

    # Data augmentation (train only)
    AUGMENT: bool = True
    NOISE_STD: float = 0.01
    JITTER_MAX: int = 10
    CHANNEL_DROPOUT_P: float = 0.05

    # Target transform
    USE_LOG1P_TARGET: bool = True  # regress log1p(length_chars)

    # Output
    OUT_DIR: str = "sentence_length_runs"
    RUN_NAME: str = "baseline_attn_pool"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_val_indices(
    n_samples: int,
    val_split: float,
    shuffle: bool,
    seed: int,
) -> Tuple[List[int], List[int]]:
    idx = list(range(n_samples))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)

    val_size = int(n_samples * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    print(f"Train size: {len(train_idx)}, val size: {len(val_idx)}")
    return train_idx, val_idx


# ----------------- Metrics -----------------

@torch.no_grad()
def compute_metrics_from_log_preds(
    y_hat_log: torch.Tensor,
    y_true_log: torch.Tensor,
    y_true_raw: torch.Tensor,
) -> Dict[str, float]:
    """Compute MAE/RMSE in both raw-char space and log space."""
    mse_log = torch.mean((y_hat_log - y_true_log) ** 2).item()
    rmse_log = math.sqrt(mse_log)
    mae_log = torch.mean(torch.abs(y_hat_log - y_true_log)).item()

    y_hat_raw = torch.expm1(y_hat_log).clamp_min(0.0)
    mae_raw = torch.mean(torch.abs(y_hat_raw - y_true_raw)).item()
    mse_raw = torch.mean((y_hat_raw - y_true_raw) ** 2).item()
    rmse_raw = math.sqrt(mse_raw)

    return {
        "mae_raw": mae_raw,
        "rmse_raw": rmse_raw,
        "mae_log": mae_log,
        "rmse_log": rmse_log,
    }


def _r2_from_sums(ss_res: float, sum_y: float, sum_y2: float, n: int) -> float:
    """Compute R^2 = 1 - SS_res / SS_tot using one-pass sums.

    SS_tot = sum(y^2) - n * mean(y)^2
    """
    if n <= 1:
        return float("nan")
    mean_y = sum_y / n
    ss_tot = max(sum_y2 - n * (mean_y ** 2), 0.0)
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - (ss_res / (ss_tot + 1e-12))


# ----------------- Train / eval loops -----------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_log1p_target: bool,
) -> Dict[str, float]:
    model.train()
    loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    total_n = 0

    sum_mae_raw = 0.0
    sum_rmse_raw_sq = 0.0
    sum_mae_log = 0.0
    sum_rmse_log_sq = 0.0

    # R^2 accumulators (raw chars)
    ss_res_raw = 0.0
    sum_y_raw = 0.0
    sum_y_raw2 = 0.0

    for batch in loader:
        eeg = batch["eeg"].to(device)                 # (B, C, T)
        subj = batch["subject_id"].to(device)         # (B,)
        y = batch["target"].to(device).float()        # (B,) log1p or raw

        optimizer.zero_grad()
        y_hat_log = model(eeg, subj)                  # (B,) predicted log1p(chars)

        if use_log1p_target:
            y_true_log = y
            loss = loss_fn(y_hat_log, y_true_log)
            y_true_raw = torch.expm1(y_true_log).clamp_min(0.0)
        else:
            y_true_raw = y
            loss = loss_fn(torch.expm1(y_hat_log).clamp_min(0.0), y_true_raw)
            y_true_log = torch.log1p(y_true_raw)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        B = eeg.size(0)
        total_loss += loss.item() * B
        total_n += B

        m = compute_metrics_from_log_preds(
            y_hat_log=y_hat_log.detach(),
            y_true_log=y_true_log.detach(),
            y_true_raw=y_true_raw.detach(),
        )
        sum_mae_raw += m["mae_raw"] * B
        sum_rmse_raw_sq += (m["rmse_raw"] ** 2) * B
        sum_mae_log += m["mae_log"] * B
        sum_rmse_log_sq += (m["rmse_log"] ** 2) * B

        # R^2 sums in raw space
        y_hat_raw = torch.expm1(y_hat_log.detach()).clamp_min(0.0)
        ss_res_raw += torch.sum((y_hat_raw - y_true_raw.detach()) ** 2).item()
        sum_y_raw += torch.sum(y_true_raw.detach()).item()
        sum_y_raw2 += torch.sum((y_true_raw.detach()) ** 2).item()

    r2_raw = _r2_from_sums(ss_res_raw, sum_y_raw, sum_y_raw2, total_n)

    return {
        "loss": total_loss / total_n,
        "mae_raw": sum_mae_raw / total_n,
        "rmse_raw": math.sqrt(sum_rmse_raw_sq / total_n),
        "mae_log": sum_mae_log / total_n,
        "rmse_log": math.sqrt(sum_rmse_log_sq / total_n),
        "r2_raw": r2_raw,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_log1p_target: bool,
) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    total_n = 0

    sum_mae_raw = 0.0
    sum_rmse_raw_sq = 0.0
    sum_mae_log = 0.0
    sum_rmse_log_sq = 0.0

    # R^2 accumulators (raw chars)
    ss_res_raw = 0.0
    sum_y_raw = 0.0
    sum_y_raw2 = 0.0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        y = batch["target"].to(device).float()

        y_hat_log = model(eeg, subj)

        if use_log1p_target:
            y_true_log = y
            loss = loss_fn(y_hat_log, y_true_log)
            y_true_raw = torch.expm1(y_true_log).clamp_min(0.0)
        else:
            y_true_raw = y
            loss = loss_fn(torch.expm1(y_hat_log).clamp_min(0.0), y_true_raw)
            y_true_log = torch.log1p(y_true_raw)

        B = eeg.size(0)
        total_loss += loss.item() * B
        total_n += B

        m = compute_metrics_from_log_preds(
            y_hat_log=y_hat_log,
            y_true_log=y_true_log,
            y_true_raw=y_true_raw,
        )
        sum_mae_raw += m["mae_raw"] * B
        sum_rmse_raw_sq += (m["rmse_raw"] ** 2) * B
        sum_mae_log += m["mae_log"] * B
        sum_rmse_log_sq += (m["rmse_log"] ** 2) * B

        y_hat_raw = torch.expm1(y_hat_log).clamp_min(0.0)
        ss_res_raw += torch.sum((y_hat_raw - y_true_raw) ** 2).item()
        sum_y_raw += torch.sum(y_true_raw).item()
        sum_y_raw2 += torch.sum((y_true_raw ** 2)).item()

    r2_raw = _r2_from_sums(ss_res_raw, sum_y_raw, sum_y_raw2, total_n)

    return {
        "loss": total_loss / total_n,
        "mae_raw": sum_mae_raw / total_n,
        "rmse_raw": math.sqrt(sum_rmse_raw_sq / total_n),
        "mae_log": sum_mae_log / total_n,
        "rmse_log": math.sqrt(sum_rmse_log_sq / total_n),
        "r2_raw": r2_raw,
    }


# ----------------- Main -----------------

def main():
    cfg = Config()
    if cfg.SUBJECT_CODES is None:
        cfg.SUBJECT_CODES = [f"{i:02d}" for i in range(1, 6)]

    set_seed(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Output directories
    run_dir = os.path.join(cfg.OUT_DIR, cfg.RUN_NAME)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    # 1) Load EEG + sentence lengths (raw)
    X, y_raw, sentences, subj_ids = build_sentence_length_tensors(
        chisco_root=cfg.CHISCO_ROOT,
        subject_codes=cfg.SUBJECT_CODES,
        length_json_path=cfg.SENT_LEN_PATH,
    )
    n_samples, n_channels, _ = X.shape
    n_subjects = len(cfg.SUBJECT_CODES)

    # 2) Target transform
    if cfg.USE_LOG1P_TARGET:
        y_target = torch.log1p(y_raw)
        print("Training target: log1p(length_chars)")
    else:
        y_target = y_raw
        print("Training target: raw length_chars (not recommended)")

    # 3) Train/val split
    train_idx, val_idx = make_train_val_indices(
        n_samples=n_samples,
        val_split=cfg.VAL_SPLIT,
        shuffle=cfg.SHUFFLE_DATA,
        seed=cfg.RANDOM_SEED,
    )

    # 4) Datasets
    train_dataset = ChiscoImagineDataset(
        eeg=X[train_idx],
        sentences=[sentences[i] for i in train_idx],
        subject_ids=subj_ids[train_idx],
        targets=y_target[train_idx],
        augment=cfg.AUGMENT,
        noise_std=cfg.NOISE_STD,
        jitter_max=cfg.JITTER_MAX,
        channel_dropout_p=cfg.CHANNEL_DROPOUT_P,
    )
    val_dataset = ChiscoImagineDataset(
        eeg=X[val_idx],
        sentences=[sentences[i] for i in val_idx],
        subject_ids=subj_ids[val_idx],
        targets=y_target[val_idx],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 5) Model
    model = EEGToSentenceLengthModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        latent_dim=cfg.LATENT_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        final_hidden=cfg.FINAL_HIDDEN,
        n_blocks=cfg.N_BLOCKS,
        attn_hidden=cfg.ATTN_HIDDEN,
        use_subject_embed_in_head=cfg.USE_SUBJ_EMB_IN_HEAD,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * cfg.NUM_EPOCHS
    warmup_steps = math.ceil(total_steps * cfg.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 6) Training loop
    best_val_mae = float("inf")
    hist = {
        "train_loss": [],
        "val_loss": [],
        "train_mae_raw": [],
        "val_mae_raw": [],
        "train_rmse_raw": [],
        "val_rmse_raw": [],
        "train_rmse_log": [],
        "val_rmse_log": [],
        "train_r2_raw": [],
        "val_r2_raw": [],
    }

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{cfg.NUM_EPOCHS} ===")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_log1p_target=cfg.USE_LOG1P_TARGET,
        )
        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            use_log1p_target=cfg.USE_LOG1P_TARGET,
        )

        print(
            f"Train | loss {train_stats['loss']:.4f} | MAE(chars) {train_stats['mae_raw']:.2f} "
            f"| RMSE(chars) {train_stats['rmse_raw']:.2f} | RMSE(log) {train_stats['rmse_log']:.4f} "
            f"| R^2 {train_stats['r2_raw']:.4f}"
        )
        print(
            f"Val   | loss {val_stats['loss']:.4f} | MAE(chars) {val_stats['mae_raw']:.2f} "
            f"| RMSE(chars) {val_stats['rmse_raw']:.2f} | RMSE(log) {val_stats['rmse_log']:.4f} "
            f"| R^2 {val_stats['r2_raw']:.4f}"
        )

        hist["train_loss"].append(train_stats["loss"])
        hist["val_loss"].append(val_stats["loss"])
        hist["train_mae_raw"].append(train_stats["mae_raw"])
        hist["val_mae_raw"].append(val_stats["mae_raw"])
        hist["train_rmse_raw"].append(train_stats["rmse_raw"])
        hist["val_rmse_raw"].append(val_stats["rmse_raw"])
        hist["train_rmse_log"].append(train_stats["rmse_log"])
        hist["val_rmse_log"].append(val_stats["rmse_log"])
        hist["train_r2_raw"].append(train_stats["r2_raw"])
        hist["val_r2_raw"].append(val_stats["r2_raw"])

        if val_stats["mae_raw"] < best_val_mae:
            best_val_mae = val_stats["mae_raw"]
            ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg.__dict__,
                    "best_val_mae_raw": best_val_mae,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved new best model to {ckpt_path} (val_MAE_chars={best_val_mae:.2f})")

    # 7) Save history + plots
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(hist, f, indent=2)

    epochs = list(range(1, cfg.NUM_EPOCHS + 1))

    def _plot(y_train, y_val, ylabel, title, fname):
        plt.figure()
        plt.plot(epochs, y_train, label="Train")
        plt.plot(epochs, y_val, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "plots", fname))
        plt.close()

    _plot(hist["train_loss"], hist["val_loss"], "SmoothL1 Loss (log space)", "Loss", "loss_curve.png")
    _plot(hist["train_mae_raw"], hist["val_mae_raw"], "MAE (chars)", "MAE (chars)", "mae_chars_curve.png")
    _plot(hist["train_rmse_raw"], hist["val_rmse_raw"], "RMSE (chars)", "RMSE (chars)", "rmse_chars_curve.png")
    _plot(hist["train_rmse_log"], hist["val_rmse_log"], "RMSE (log1p(chars))", "RMSE (log)", "rmse_log_curve.png")
    _plot(hist["train_r2_raw"], hist["val_r2_raw"], "R^2 (raw chars)", "R^2 (raw chars)", "r2_chars_curve.png")

    print("\nTraining finished.")
    print(f"Best val MAE (chars): {best_val_mae:.2f}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

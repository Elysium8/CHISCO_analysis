# train_semantic_genres.py
#
# Semantic genre classification on CHISCO imagined-speech EEG
# with:
#   - EEGFrontEnd + BrainModule
#   - data augmentation (train only)
#   - linear warmup + decay LR scheduler (per-batch)
#   - plotting of loss/accuracy curves

import os
import sys
import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# make sure we can import the dataset code
sys.path.append("./ds005170")

from Brain_Module import BrainModule
from data_loader import (
    load_chisco_imagine_all_subjects,
    ChiscoImagineDataset,
)
from label_matching import (
    load_text_to_class_map,
    num_classes_from_classnumber,
)
from model_parts import EEGFrontEnd


# ----------------- Config -----------------

class Config:
    # Paths
    CHISCO_ROOT = "./ds005170/derivatives/preprocessed_pkl"
    TEXTMAP_PATH = "./ds005170/json/textmaps.json"
    CLASSNUMBER_PATH = "./ds005170/json/classnumber.json"

    # Subjects
    SUBJECT_CODES = [f"{i:02d}" for i in range(1, 6)]

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT = 0.2          # 20% validation
    SHUFFLE_DATA = True
    RANDOM_SEED = 42
    WARMUP_RATIO = 0.1       # 10% of total steps for warmup

    # Model
    HIDDEN_CHANNELS = 256
    FINAL_HIDDEN = 256
    N_BLOCKS = 5
    LATENT_DIM = 256
    POOLING = "mean"         # "mean" or "max"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- Model wrapper -----------------

class BrainGenreClassifier(nn.Module):
    """
    EEGFrontEnd -> BrainModule -> pooling -> classifier
    """
    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        n_classes: int,
        latent_dim: int = 256,
        hidden_channels: int = 256,
        final_hidden: int = 256,
        n_blocks: int = 5,
        pooling: str = "mean",
    ):
        super().__init__()

        self.pooling = pooling

        # EEG front-end (EEGNet-style)
        self.eeg_frontend = EEGFrontEnd(
            in_ch=n_channels,
            temporal_kern=64,
            F1=16,
            D=2,
        )
        frontend_out_ch = 16 * 2

        # Brain module
        self.brain = BrainModule(
            in_channels=frontend_out_ch,
            out_channels=latent_dim,
            n_subjects=n_subjects,
            hidden_channels=hidden_channels,
            final_hidden=final_hidden,
            n_blocks=n_blocks,
            spatial_out_channels=frontend_out_ch,
        )

        # learnable "sensor positions" for the spatial layer
        self.sensor_pos = nn.Parameter(torch.randn(frontend_out_ch, 2) * 0.1)

        # classifier head
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.eeg_frontend(x)                           # (B, C', T')
        z = self.brain(x, self.sensor_pos, subject_ids)    # (B, latent_dim, T')

        if self.pooling == "mean":
            z_pooled = z.mean(dim=-1)
        elif self.pooling == "max":
            z_pooled, _ = z.max(dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.classifier(z_pooled)                 # (B, n_classes)
        return logits


# ----------------- Data prep -----------------

def build_genre_tensors(cfg: Config):
    """
    Load CHISCO imagined-speech EEG and map sentences to
    semantic genre IDs using textmaps.json.

    Returns:
        X_kept:        (N, C, T) tensor
        sentences_kept:list[str] length N
        subj_ids_kept: (N,) long tensor
        genre_targets: (N,) long tensor of class IDs
    """
    print("Loading CHISCO EEG + sentences...")
    X, sentences, subj_ids = load_chisco_imagine_all_subjects(
        root=cfg.CHISCO_ROOT,
        subject_codes=cfg.SUBJECT_CODES,
    )
    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints.")

    print("Loading sentence -> semantic class map...")
    text2class = load_text_to_class_map(cfg.TEXTMAP_PATH)

    genre_labels: List[int] = []
    keep_indices: List[int] = []
    missing = 0

    for i, sent in enumerate(sentences):
        cls = text2class.get(sent)
        if cls is None:
            missing += 1
            continue
        genre_labels.append(cls)
        keep_indices.append(i)

    if missing > 0:
        print(f"Warning: {missing} sentences not found in textmaps.json and were skipped.")
    print(f"Using {len(keep_indices)} trials with valid genre labels.")

    X_kept = X[keep_indices]                               # (N, C, T)
    subj_ids_kept = subj_ids[keep_indices]                 # (N,)
    sentences_kept = [sentences[i] for i in keep_indices]  # list[str]
    genre_targets = torch.tensor(genre_labels, dtype=torch.long)

    return X_kept, sentences_kept, subj_ids_kept, genre_targets


def make_train_val_indices(
    n_samples: int,
    val_split: float,
    shuffle: bool,
    seed: int,
):
    idx = list(range(n_samples))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)

    val_size = int(n_samples * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    print(f"Train size: {len(train_idx)}, val size: {len(val_idx)}")
    return train_idx, val_idx


# ----------------- Training / evaluation -----------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device)            # (B, C, T)
        subj = batch["subject_id"].to(device)    # (B,)
        target = batch["target"].to(device)      # (B,)

        optimizer.zero_grad()
        logits = model(eeg, subj)                # (B, n_classes)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()                     # per-batch LR step

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * eeg.size(0)
        total_correct += (preds == target).sum().item()
        total += eeg.size(0)

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / total
            avg_acc = total_correct / total * 100.0
            print(
                f"Epoch {epoch} | Step {step+1}/{len(loader)} "
                f"| Loss {avg_loss:.4f} | Acc {avg_acc:.2f}%"
            )

    return total_loss / total, (total_correct / total) * 100.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        target = batch["target"].to(device)

        logits = model(eeg, subj)
        loss = criterion(logits, target)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * eeg.size(0)
        total_correct += (preds == target).sum().item()
        total += eeg.size(0)

    return total_loss / total, (total_correct / total) * 100.0


# ----------------- Main -----------------

def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load tensors + labels
    X, sentences, subj_ids, targets = build_genre_tensors(cfg)
    n_samples, n_channels, _ = X.shape

    # 2) Train/val split (indices)
    train_idx, val_idx = make_train_val_indices(
        n_samples=n_samples,
        val_split=cfg.VAL_SPLIT,
        shuffle=cfg.SHUFFLE_DATA,
        seed=cfg.RANDOM_SEED,
    )

    # 3) Build train/val datasets
    train_dataset = ChiscoImagineDataset(
        eeg=X[train_idx],
        sentences=[sentences[i] for i in train_idx],
        subject_ids=subj_ids[train_idx],
        targets=targets[train_idx],
        augment=True,               # ★ AUGMENTATION ON FOR TRAIN ★
        noise_std=0.01,
        jitter_max=10,
        channel_dropout_p=0.05,
    )

    val_dataset = ChiscoImagineDataset(
        eeg=X[val_idx],
        sentences=[sentences[i] for i in val_idx],
        subject_ids=subj_ids[val_idx],
        targets=targets[val_idx],
        augment=False,              # ★ NO AUGMENTATION FOR VAL ★
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

    # 4) Model + optimizer + LR scheduler
    n_subjects = len(cfg.SUBJECT_CODES)
    n_classes = num_classes_from_classnumber(cfg.CLASSNUMBER_PATH)

    print(f"n_channels={n_channels}, n_subjects={n_subjects}, n_classes={n_classes}")

    model = BrainGenreClassifier(
        n_channels=n_channels,
        n_subjects=n_subjects,
        n_classes=n_classes,
        latent_dim=cfg.LATENT_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        final_hidden=cfg.FINAL_HIDDEN,
        n_blocks=cfg.N_BLOCKS,
        pooling=cfg.POOLING,
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

    # 5) Training loop with history
    best_val_acc = 0.0
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{cfg.NUM_EPOCHS} ===")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        print(f"Train   | Loss {train_loss:.4f} | Acc {train_acc:.2f}%")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val     | Loss {val_loss:.4f} | Acc {val_acc:.2f}%")

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", "best_genre_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg.__dict__,
                    "best_val_acc": best_val_acc,
                },
                ckpt_path,
            )
            print(f"Saved new best model to {ckpt_path} (val_acc={best_val_acc:.2f}%)")

    # 6) Plot curves
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, cfg.NUM_EPOCHS + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, train_loss_hist, label="Train Loss")
    plt.plot(epochs, val_loss_hist, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs (semantic genres, imagine)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/loss_curve_semantic_imagine.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_acc_hist, label="Train Acc")
    plt.plot(epochs, val_acc_hist, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over epochs (semantic genres, imagine)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/accuracy_curve_semantic_imagine.png")
    plt.close()

    print(f"\nTraining finished. Best val accuracy: {best_val_acc:.2f}%")
    print("Saved plots to:")
    print("  plots/loss_curve_semantic_imagine.png")
    print("  plots/accuracy_curve_semantic_imagine.png")


if __name__ == "__main__":
    main()

import os
import math
import json
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from Brain_Module import BrainModule
from data_loader import (
    load_chisco_imagine_all_subjects,
    ChiscoImagineDataset,
)


# ----------------- Config -----------------

class Config:
    # Paths (relative to this script)
    CHISCO_ROOT = "./ds005170/derivatives/preprocessed_pkl"
    QUESTIONMAP_PATH = "./ds005170/json/question_statement_mapping.json"

    # Subjects: 01..23 (adjust if needed)
    SUBJECT_CODES = [f"{i:02d}" for i in range(1, 6)]

    # Training params
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    VAL_SPLIT = 0.2
    SHUFFLE_DATA = True
    RANDOM_SEED = 42
    WARMUP_RATIO = 0.1

    # Model architecture
    HIDDEN_CHANNELS = 256
    FINAL_HIDDEN = 256
    N_BLOCKS = 5
    LATENT_DIM = 256
    POOLING = "mean"


# ----------------- Utility: Seeding -----------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- Model wrapper: Brain + classifier -----------------

class BrainSentenceTypeClassifier(nn.Module):
    """
    Wraps BrainModule with:
      - Learnable sensor positions
      - Temporal pooling
      - Final linear classifier for question vs statement (2 classes)
    """
    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        n_classes: int = 2,
        latent_dim: int = 256,
        hidden_channels: int = 256,
        final_hidden: int = 256,
        n_blocks: int = 5,
        pooling: str = "mean",
    ):
        super().__init__()

        self.pooling = pooling

        self.brain = BrainModule(
            in_channels=n_channels,
            out_channels=latent_dim,
            n_subjects=n_subjects,
            hidden_channels=hidden_channels,
            final_hidden=final_hidden,
            n_blocks=n_blocks,
            spatial_out_channels=n_channels,
        )

        # Learnable sensor positions (C, 2) – replace later with real montage if you want
        self.sensor_pos = nn.Parameter(
            torch.randn(n_channels, 2) * 0.1
        )

        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, C, T)
        subject_ids: (B,)
        returns:
            logits: (B, n_classes)
        """
        z = self.brain(x, self.sensor_pos, subject_ids)  # (B, latent_dim, T)

        if self.pooling == "mean":
            z_pooled = z.mean(dim=-1)   # (B, latent_dim)
        elif self.pooling == "max":
            z_pooled, _ = z.max(dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.classifier(z_pooled)
        return logits


# ----------------- Data prep: question vs statement -----------------

def build_sentence_type_dataset(cfg: Config) -> Tuple[ChiscoImagineDataset, List[int]]:
    """
    Loads CHISCO imagined EEG and maps each sentence to:
        0 = statement
        1 = question
    using question_statement_mapping.json.

    Returns:
        dataset: ChiscoImagineDataset with targets = 0/1
        keep_indices: list of indices kept (for debugging)
    """
    print("Loading CHISCO EEG + sentences...")
    X, sentences, subj_ids = load_chisco_imagine_all_subjects(
        root=cfg.CHISCO_ROOT,
        subject_codes=cfg.SUBJECT_CODES,
    )
    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints.")

    print("Loading sentence -> question/statement map from JSON...")
    with open(cfg.QUESTIONMAP_PATH, "r", encoding="utf-8") as f:
        sent2qs = json.load(f)   # {sentence: 0 or 1}

    type_labels: List[int] = []
    keep_indices: List[int] = []

    missing = 0
    for i, sent in enumerate(sentences):
        lbl = sent2qs.get(sent)
        if lbl is None:
            missing += 1
            continue
        type_labels.append(int(lbl))  # ensure 0/1 int
        keep_indices.append(i)

    if missing > 0:
        print(f"Warning: {missing} sentences not found in question map and were skipped.")
    print(f"Using {len(keep_indices)} trials with valid sentence-type labels.")

    type_targets = torch.tensor(type_labels, dtype=torch.long)
    X_kept = X[keep_indices]
    subj_ids_kept = subj_ids[keep_indices]
    sentences_kept = [sentences[i] for i in keep_indices]

    dataset = ChiscoImagineDataset(
        eeg=X_kept,
        sentences=sentences_kept,
        subject_ids=subj_ids_kept,
        targets=type_targets,
    )

    return dataset, keep_indices


# ----------------- Train/val split -----------------

def make_train_val_splits(
    dataset: ChiscoImagineDataset,
    val_split: float,
    shuffle: bool,
    seed: int,
) -> Tuple[Subset, Subset]:
    n = len(dataset)
    indices = list(range(n))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(indices)

    val_size = int(n * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    print(f"Train size: {len(train_set)}, val size: {len(val_set)}")
    return train_set, val_set


# ----------------- Training / evaluation loops -----------------

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

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device)               # (B, C, T)
        subj = batch["subject_id"].to(device)       # (B,)
        targets = batch["target"].to(device)        # (B,) 0/1

        optimizer.zero_grad()

        logits = model(eeg, subj)                   # (B, 2)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * eeg.size(0)

        preds = logits.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        running_total += eeg.size(0)

        if (step + 1) % 50 == 0:
            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total * 100.0
            print(
                f"Epoch {epoch} | Step {step+1}/{len(loader)} "
                f"| Loss {avg_loss:.4f} | Acc {avg_acc:.2f}%"
            )

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        targets = batch["target"].to(device)

        logits = model(eeg, subj)
        loss = criterion(logits, targets)

        running_loss += loss.item() * eeg.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        running_total += eeg.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total * 100.0
    return epoch_loss, epoch_acc


# ----------------- Main -----------------

def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Build dataset with question vs statement labels
    dataset, kept = build_sentence_type_dataset(cfg)

    # 2) Train/val split
    train_set, val_set = make_train_val_splits(
        dataset,
        val_split=cfg.VAL_SPLIT,
        shuffle=cfg.SHUFFLE_DATA,
        seed=cfg.RANDOM_SEED,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3) Model + optimizer + scheduler
    n_channels = dataset.eeg.shape[1]
    n_subjects = len(cfg.SUBJECT_CODES)
    n_classes = 2  # question vs statement

    print(f"n_channels={n_channels}, n_subjects={n_subjects}, n_classes={n_classes}")

    model = BrainSentenceTypeClassifier(
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

    # 4) Training loop
    best_val_acc = 0.0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{cfg.NUM_EPOCHS} ===")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        print(f"Train   | Loss {train_loss:.4f} | Acc {train_acc:.2f}%")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val     | Loss {val_loss:.4f} | Acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", "best_sentence_type_model.pt")
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

    print(f"\nTraining finished. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

# train_semantic_genres.py  (updated with scheduler + plotting)

import os
import random
import math
from typing import List, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append("./ds005170")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup   # NEW

from Brain_Module import BrainModule
from data_loader import (
    load_chisco_imagine_all_subjects,
    ChiscoImagineDataset,
)
from label_matching import (
    load_text_to_class_map,
    load_class_id_to_name_map,
    num_classes_from_classnumber,
)
from augmentations import EEGAugment
from contrastive import ContrastiveDataset, NTXentLoss, SimCLR

# ----------------- Config -----------------

class Config:
    CHISCO_ROOT = "./ds005170/derivatives/preprocessed_pkl"
    TEXTMAP_PATH = "./ds005170/json/textmaps.json"
    CLASSNUMBER_PATH = "./ds005170/json/classnumber.json"

    SUBJECT_CODES = [f"{i:02d}" for i in range(1, 6)]

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT = 0.2
    SHUFFLE_DATA = True
    RANDOM_SEED = 42

    HIDDEN_CHANNELS = 256
    FINAL_HIDDEN = 256
    N_BLOCKS = 5
    LATENT_DIM = 256
    POOLING = "mean"

    WARMUP_RATIO = 0.1   # NEW

    PRETRAIN_EPOCHS = 30
    PRETRAIN_LR = 1e-3
    TEMP = 0.6


def pretrain_simclr(cfg, base_dataset, brain_module, sensor_pos, device):  # Added sensor_pos
    """
    Runs self-supervised contrastive learning on the BrainModule.
    Modifies brain_module weights in-place.
    """
    print("\n=== Starting SimCLR Pre-training ===")

    # 1. Setup Augmentation & Dataset
    augmenter = EEGAugment(noise_std=0.02, p_dropout=0.4, max_channel_drop=2)
    contrastive_set = ContrastiveDataset(base_dataset, augmenter)

    train_loader = DataLoader(contrastive_set, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # 2. Setup SimCLR Wrapper
    simclr_model = SimCLR(brain_module, latent_dim=cfg.LATENT_DIM).to(device)

    optimizer = torch.optim.AdamW(simclr_model.parameters(), lr=cfg.PRETRAIN_LR)
    criterion = NTXentLoss(temperature=cfg.TEMP)

    # 3. Training Loop
    simclr_model.train()

    for epoch in range(1, cfg.PRETRAIN_EPOCHS + 1):
        total_loss = 0
        steps = 0

        for batch in train_loader:
            v1 = batch["view1"].to(device)
            v2 = batch["view2"].to(device)
            subj = batch["subject_id"].to(device)

            optimizer.zero_grad()

            # Forward pass using the passed sensor_pos
            # We must ensure sensor_pos is on the correct device
            _, z1 = simclr_model(v1, sensor_pos, subj)
            _, z2 = simclr_model(v2, sensor_pos, subj)

            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"Pretrain Epoch {epoch}/{cfg.PRETRAIN_EPOCHS} | NT-Xent Loss: {avg_loss:.4f}")

    print("=== Pre-training Complete ===\n")

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- Model wrapper -----------------

class BrainGenreClassifier(nn.Module):
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

        self.brain = BrainModule(
            in_channels=n_channels,
            out_channels=latent_dim,
            n_subjects=n_subjects,
            hidden_channels=hidden_channels,
            final_hidden=final_hidden,
            n_blocks=n_blocks,
            spatial_out_channels=n_channels,
        )

        self.sensor_pos = nn.Parameter(torch.randn(n_channels, 2) * 0.1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x, subject_ids):
        z = self.brain(x, self.sensor_pos, subject_ids)

        if self.pooling == "mean":
            z_pooled = z.mean(dim=-1)
        else:
            z_pooled, _ = z.max(dim=-1)

        return self.classifier(z_pooled)


# ----------------- Data prep -----------------

def build_genre_dataset(cfg: Config):
    print("Loading CHISCO EEG + sentences...")
    X, sentences, subj_ids = load_chisco_imagine_all_subjects(
        root=cfg.CHISCO_ROOT,
        subject_codes=cfg.SUBJECT_CODES,
    )
    print(f"Loaded {X.shape[0]} trials.")

    text2class = load_text_to_class_map(cfg.TEXTMAP_PATH)

    genre_labels = []
    keep_indices = []
    missing = 0

    for i, sent in enumerate(sentences):
        cls = text2class.get(sent)
        if cls is None:
            missing += 1
            continue
        genre_labels.append(cls)
        keep_indices.append(i)

    if missing > 0:
        print(f"Warning: {missing} missing labels.")

    dataset = ChiscoImagineDataset(
        eeg=X[keep_indices],
        sentences=[sentences[i] for i in keep_indices],
        subject_ids=subj_ids[keep_indices],
        targets=torch.tensor(genre_labels, dtype=torch.long),
    )
    return dataset, keep_indices


def make_train_val_splits(dataset, val_split, shuffle, seed):
    n = len(dataset)
    idx = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(idx)

    val_size = int(n * val_split)
    return (
        Subset(dataset, idx[val_size:]),
        Subset(dataset, idx[:val_size])
    )


# ----------------- Train one epoch -----------------

def train_one_epoch(model, loader, optimizer, device, epoch):
    criterion = nn.CrossEntropyLoss()
    model.train()

    total_loss = 0
    total_correct = 0
    total = 0

    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(eeg, subj)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * eeg.size(0)
        total_correct += (preds == target).sum().item()
        total += eeg.size(0)

    return total_loss / total, (total_correct / total) * 100


# ----------------- Evaluate -----------------

@torch.no_grad()
def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0
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

    return total_loss / total, (total_correct / total) * 100


# ----------------- Main -----------------

def main():
    cfg = Config()
    # Assuming set_seed is defined globally or imported
    # set_seed(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # build_genre_dataset must return the raw dataset without splits yet
    dataset, _ = build_genre_dataset(cfg)

    n_channels = dataset.eeg.shape[1]
    n_subjects = len(cfg.SUBJECT_CODES)
    n_classes = num_classes_from_classnumber(cfg.CLASSNUMBER_PATH)

    # 2. Initialize Full Model
    # We initialize the full classifier now so we have the BrainModule ready.
    full_model = BrainGenreClassifier(
        n_channels=n_channels,
        n_subjects=n_subjects,
        n_classes=n_classes,
        latent_dim=cfg.LATENT_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        final_hidden=cfg.FINAL_HIDDEN,
        n_blocks=cfg.N_BLOCKS,
        pooling=cfg.POOLING
    ).to(device)

    # ---------------------------------------------------------
    # PHASE 1: Unsupervised Pre-training (SimCLR)
    # ---------------------------------------------------------
    # We extract the .brain submodule and train it using contrastive loss.
    # The weights inside full_model.brain will be updated.
    pretrain_simclr(cfg, dataset, full_model.brain, full_model.sensor_pos, device)

    # ---------------------------------------------------------
    # PHASE 2: Supervised Fine-tuning
    # ---------------------------------------------------------
    print("=== Starting Supervised Fine-tuning ===")

    # Prepare Splits
    train_set, val_set = make_train_val_splits(dataset, cfg.VAL_SPLIT, cfg.SHUFFLE_DATA, cfg.RANDOM_SEED)
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Re-initialize Optimizer for the full model (Classifier + Finetuning Encoder)
    # We typically use a lower learning rate for finetuning than pre-training
    optimizer = torch.optim.AdamW(
        full_model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # Setup Scheduler
    total_steps = len(train_loader) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Metric tracking
    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []
    best_val_acc = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS}")

        # Train step (Classification Loss)
        train_loss, train_acc = train_one_epoch(full_model, train_loader, optimizer, device, epoch)

        # Validation step
        val_loss, val_acc = evaluate(full_model, val_loader, device)

        # Update LR
        scheduler.step()

        # Logging
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(full_model.state_dict(), "checkpoints/best_genre_model_finetuned.pt")
            print(">> Saved new best model.")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    os.makedirs("plots", exist_ok=True)
    epochs_range = range(1, cfg.NUM_EPOCHS + 1)

    # Loss Plot
    plt.figure()
    plt.plot(epochs_range, train_loss_hist, label="Train Loss")
    plt.plot(epochs_range, val_loss_hist, label="Val Loss")
    plt.legend()
    plt.title("Fine-tuning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plots/finetune_loss.png")
    plt.close()

    # Accuracy Plot
    plt.figure()
    plt.plot(epochs_range, train_acc_hist, label="Train Acc")
    plt.plot(epochs_range, val_acc_hist, label="Val Acc")
    plt.legend()
    plt.title("Fine-tuning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig("plots/finetune_accuracy.png")
    plt.close()

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
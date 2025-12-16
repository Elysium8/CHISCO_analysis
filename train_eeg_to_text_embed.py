# train_eeg_to_text_embed.py
#
# Train EEG→text embedding model on CHISCO imagined-speech data
# using precomputed sentence embeddings and a contrastive loss.

import os
import sys
import math
import random
from typing import List, Tuple
from simple_experiment_logger import SimpleExperimentLogger

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# make sure we can import ds005170 modules
sys.path.append("./ds005170")

from Brain_Module import BrainModule
from data_loader import (
    load_chisco_imagine_all_subjects,load_chisco_read_all_subjects,
    ChiscoImagineDataset,
)
from model_parts import EEGFrontEnd


# ----------------- Config -----------------

class Config:
    # Paths
    CHISCO_ROOT = "./ds005170/derivatives/preprocessed_pkl"
    SENT_EMB_PATH = "./ds005170/json/sentence_bert_embs.json"  # <-- your precomputed embeddings

    TASK = "read"  # "read" or "imagine"
    SPLIT_MODE = "unseen_sentence"  # "iid" | "unseen_subject" | "unseen_sentence"
    HOLDOUT_SUBJECTS = ["05"]  # used when SPLIT_MODE="unseen_subject"
    HOLDOUT_SENT_FRACTION = 0.2  # used when SPLIT_MODE="unseen_sentence"


    SUBJECT_CODES = [f"{i:02d}" for i in range(1, 6)]

    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT = 0.2
    SHUFFLE_DATA = True
    RANDOM_SEED = 42
    WARMUP_RATIO = 0.1
    TEMPERATURE = 0.07   # contrastive temperature

    # Model
    HIDDEN_CHANNELS = 256
    FINAL_HIDDEN = 256
    N_BLOCKS = 5
    LATENT_DIM = 256
    POOLING = "mean"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- Model: EEG -> text embedding -----------------

class EEGToTextEmbedModel(nn.Module):
    """
    EEGFrontEnd -> BrainModule -> pooling -> linear projection
    Outputs L2-normalized embeddings in the same dimension as text embeddings.
    """
    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        text_dim: int,
        latent_dim: int = 256,
        hidden_channels: int = 256,
        final_hidden: int = 256,
        n_blocks: int = 5,
        pooling: str = "mean",
    ):
        super().__init__()

        self.pooling = pooling

        # EEG front-end
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

        # Learnable sensor positions
        self.sensor_pos = nn.Parameter(torch.randn(frontend_out_ch, 2) * 0.1)

        # Projection into text embedding space
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, text_dim)
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        subject_ids: (B,)
        returns:
            e: (B, text_dim) L2-normalized EEG embeddings
        """
        # EEG front-end
        x = self.eeg_frontend(x)                       # (B, C', T')

        # Brain module
        z = self.brain(x, self.sensor_pos, subject_ids)  # (B, latent_dim, T')

        # Temporal pooling
        if self.pooling == "mean":
            z_pooled = z.mean(dim=-1)                 # (B, latent_dim)
        elif self.pooling == "max":
            z_pooled, _ = z.max(dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Project to text embedding space
        e = self.proj(z_pooled)                       # (B, text_dim)
        e = nn.functional.normalize(e, dim=-1)        # L2 norm
        return e


def make_split_indices(cfg, sentences, subj_ids):
    """
    Returns: train_idx, val_idx, split_info
    """
    n = len(sentences)

    if cfg.SPLIT_MODE == "iid":
        train_idx, val_idx = make_train_val_indices(n, cfg.VAL_SPLIT, cfg.SHUFFLE_DATA, cfg.RANDOM_SEED)
        split_info = {"split_mode": "iid", "val_split": cfg.VAL_SPLIT, "seed": cfg.RANDOM_SEED}
        return train_idx, val_idx, split_info

    elif cfg.SPLIT_MODE == "unseen_subject":
        code_to_idx = {code: i for i, code in enumerate(cfg.SUBJECT_CODES)}
        holdout_ids = {code_to_idx[c] for c in cfg.HOLDOUT_SUBJECTS}

        val_idx = [i for i in range(n) if int(subj_ids[i]) in holdout_ids]
        train_idx = [i for i in range(n) if int(subj_ids[i]) not in holdout_ids]
        print(f"Held out subjects {cfg.HOLDOUT_SUBJECTS} -> val trials: {len(val_idx)}")

        split_info = {
            "split_mode": "unseen_subject",
            "holdout_subjects": list(cfg.HOLDOUT_SUBJECTS),
            "holdout_subject_indices": sorted(list(holdout_ids)),
            "seed": cfg.RANDOM_SEED,
        }
        return train_idx, val_idx, split_info

    elif cfg.SPLIT_MODE == "unseen_sentence":
        rng = random.Random(cfg.RANDOM_SEED)
        unique_sents = sorted(set(sentences))
        rng.shuffle(unique_sents)

        n_holdout = max(1, int(len(unique_sents) * cfg.HOLDOUT_SENT_FRACTION))
        holdout_sents = set(unique_sents[:n_holdout])

        val_idx = [i for i, s in enumerate(sentences) if s in holdout_sents]
        train_idx = [i for i, s in enumerate(sentences) if s not in holdout_sents]
        print(f"Held out {n_holdout}/{len(unique_sents)} sentences -> val trials: {len(val_idx)}")

        split_info = {
            "split_mode": "unseen_sentence",
            "holdout_fraction": cfg.HOLDOUT_SENT_FRACTION,
            "holdout_count": n_holdout,
            "seed": cfg.RANDOM_SEED,
        }
        return train_idx, val_idx, split_info

    else:
        raise ValueError(f"Unknown SPLIT_MODE: {cfg.SPLIT_MODE}")

# ----------------- Data prep: EEG + text embeddings -----------------

def load_sentence_embeddings(path: str):
    import json
    with open(path, "r", encoding="utf-8") as f:
        sent2emb = json.load(f)  # {sentence: [float, ...]}
    return sent2emb


def build_eeg_text_embed_tensors(cfg: Config):
    """
    Load CHISCO imagine EEG + sentences and map each sentence
    to a precomputed text embedding vector.

    Returns:
        X_kept:        (N, C, T) float32 tensor
        sentences_kept:list[str] length N
        subj_ids_kept: (N,) long tensor
        text_targets:  (N, D) float32 tensor of text embeddings
    """
    print("Loading CHISCO ", cfg.TASK, " EEG + sentences...")
    if cfg.TASK == "read":
        X, sentences, subj_ids = load_chisco_read_all_subjects(root=cfg.CHISCO_ROOT, subject_codes=cfg.SUBJECT_CODES)
    elif cfg.TASK == "imagine":
        X, sentences, subj_ids = load_chisco_imagine_all_subjects(root=cfg.CHISCO_ROOT, subject_codes=cfg.SUBJECT_CODES)
    else:
        raise ValueError("TASK must be 'read' or 'imagine'")
    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints.")

    print("Loading precomputed sentence embeddings...")
    sent2emb = load_sentence_embeddings(cfg.SENT_EMB_PATH)

    text_targets_list = []
    keep_indices = []
    missing = 0

    for i, sent in enumerate(sentences):
        emb = sent2emb.get(sent)
        if emb is None:
            missing += 1
            continue
        text_targets_list.append(torch.tensor(emb, dtype=torch.float32))
        keep_indices.append(i)

    if missing > 0:
        print(f"Warning: {missing} sentences not found in SENT_EMB_PATH and were skipped.")
    print(f"Using {len(keep_indices)} trials with valid text embeddings.")

    text_targets = torch.stack(text_targets_list, dim=0)  # (N, D)
    X_kept = X[keep_indices]                              # (N, C, T)
    subj_ids_kept = subj_ids[keep_indices]                # (N,)
    sentences_kept = [sentences[i] for i in keep_indices]

    return X_kept, sentences_kept, subj_ids_kept, text_targets


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


# ----------------- Contrastive loss (InfoNCE/CLIP style) -----------------

def contrastive_loss(
    eeg_emb: torch.Tensor,   # (B, D), normalized
    text_emb: torch.Tensor,  # (B, D), NOT necessarily normalized yet
    temperature: float = 0.07,
):
    """
    Simple symmetric InfoNCE-style loss:
        - normalize both embeddings
        - logits = (eeg_emb @ text_emb.T) / tau   (B,B)
        - each EEG sample must match its own text embedding
    """
    eeg_emb = nn.functional.normalize(eeg_emb, dim=-1)
    text_emb = nn.functional.normalize(text_emb, dim=-1)

    logits = eeg_emb @ text_emb.T / temperature   # (B, B)
    labels = torch.arange(eeg_emb.size(0), device=eeg_emb.device)

    # EEG -> text
    loss_i = nn.functional.cross_entropy(logits, labels)
    # text -> EEG (optional symmetric term)
    loss_t = nn.functional.cross_entropy(logits.T, labels)

    loss = (loss_i + loss_t) / 2.0
    return loss, logits


# ----------------- Training / evaluation -----------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    temperature: float,
):
    model.train()

    total_loss = 0.0
    total = 0

    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device)                    # (B, C, T)
        subj = batch["subject_id"].to(device)            # (B,)
        text = batch["target"].to(device)                # (B, D)

        optimizer.zero_grad()

        eeg_emb = model(eeg, subj)                       # (B, D)
        loss, logits = contrastive_loss(eeg_emb, text, temperature)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * eeg.size(0)
        total += eeg.size(0)

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / total
            print(
                f"Epoch {epoch} | Step {step+1}/{len(loader)} "
                f"| Loss {avg_loss:.4f}"
            )

    epoch_loss = total_loss / total
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
):
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

        # Retrieval metrics within the batch: EEG->text
        # logits: (B, B), true index = diagonal
        targets = torch.arange(B, device=device)
        # top-k over rows
        _, topk = logits.topk(k=min(10, B), dim=1)   # (B, k)
        correct = topk.eq(targets.unsqueeze(1))      # (B, k)

        acc1 = correct[:, :1].any(dim=1).float().mean().item() * 100.0
        acc5 = correct[:, :5].any(dim=1).float().mean().item() * 100.0 if B >= 5 else acc1
        acc10 = correct[:, :10].any(dim=1).float().mean().item() * 100.0 if B >= 10 else acc5

        total_acc1 += acc1 * B
        total_acc5 += acc5 * B
        total_acc10 += acc10 * B

    epoch_loss = total_loss / total
    epoch_acc1 = total_acc1 / total
    epoch_acc5 = total_acc5 / total
    epoch_acc10 = total_acc10 / total

    return epoch_loss, epoch_acc1, epoch_acc5, epoch_acc10


# ----------------- Main -----------------

def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load EEG + text embedding targets
    X, sentences, subj_ids, text_targets = build_eeg_text_embed_tensors(cfg)
    n_samples, n_channels, _ = X.shape
    text_dim = text_targets.shape[1]
    print(f"text_dim = {text_dim}")

    # 2) Train/val split
    train_idx, val_idx, split_info = make_split_indices(cfg, sentences, subj_ids)

    split_mode = split_info["split_mode"]  # "iid" | "unseen_subject" | "unseen_sentence"
    task = cfg.TASK  # "read" | "imagine"

    logger = SimpleExperimentLogger(
        root="experiments",
        task_name=os.path.join("eeg2text", task, split_mode),
        config={k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
        split_info=split_info,
        metric_name="val_acc10",
        higher_is_better=True,
        run_tag=None,  # optional, since task+split are now in folders
    )
    print("Logging to:", logger.run_dir)

    # 3) Build datasets
    train_dataset = ChiscoImagineDataset(
        eeg=X[train_idx],
        sentences=[sentences[i] for i in train_idx],
        subject_ids=subj_ids[train_idx],
        targets=text_targets[train_idx],
        augment=True,
        noise_std=0.01,
        jitter_max=10,
        channel_dropout_p=0.05,
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

    # 4) Model + optimizer + scheduler
    n_subjects = len(cfg.SUBJECT_CODES)

    model = EEGToTextEmbedModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        text_dim=text_dim,
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

    # 5) Training loop
    best_val_acc10 = -1e9

    train_loss_hist = []
    val_loss_hist = []
    val_acc1_hist = []
    val_acc5_hist = []
    val_acc10_hist = []

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{cfg.NUM_EPOCHS} ===")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, cfg.TEMPERATURE
        )
        print(f"Train   | Loss {train_loss:.4f}")

        val_loss, val_acc1, val_acc5, val_acc10 = evaluate(
            model, val_loader, device, cfg.TEMPERATURE
        )
        print(
            f"Val     | Loss {val_loss:.4f} | Acc@1 {val_acc1:.2f}% "
            f"| Acc@5 {val_acc5:.2f}% | Acc@10 {val_acc10:.2f}%"
        )

        # ---- LOGGER: record metrics for this epoch ----
        logger.log_epoch(
            epoch=epoch,
            train_stats={"loss": float(train_loss)},
            val_stats={
                "loss": float(val_loss),
                "acc1": float(val_acc1),
                "acc5": float(val_acc5),
                "acc10": float(val_acc10),
            },
        )

        # ---- LOGGER: save last checkpoint every epoch ----
        logger.save_last(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg.__dict__,
                "split_info": split_info,
                "train": {"loss": float(train_loss)},
                "val": {
                    "loss": float(val_loss),
                    "acc1": float(val_acc1),
                    "acc5": float(val_acc5),
                    "acc10": float(val_acc10),
                },
            }
        )

        # ---- LOGGER: save best checkpoint when Acc@10 improves ----
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
                    "config": cfg.__dict__,
                    "split_info": split_info,
                    "best_val_acc10": float(best_val_acc10),
                    "train": {"loss": float(train_loss)},
                    "val": {
                        "loss": float(val_loss),
                        "acc1": float(val_acc1),
                        "acc5": float(val_acc5),
                        "acc10": float(val_acc10),
                    },
                },
            )

            if did_save:
                print(
                    f"Saved NEW BEST -> {logger.run_dir}/best_checkpoint.pt "
                    f"(val_acc@10={best_val_acc10:.2f}%)"
                )

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_acc1_hist.append(val_acc1)
        val_acc5_hist.append(val_acc5)
        val_acc10_hist.append(val_acc10)

    # 6) Plot loss and metrics (unchanged)
    plots_dir = os.path.join(logger.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, cfg.NUM_EPOCHS + 1)

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

    print(f"\nTraining finished. Best val Acc@10: {best_val_acc10:.2f}%")
    print("Run directory:", logger.run_dir)

if __name__ == "__main__":
    main()

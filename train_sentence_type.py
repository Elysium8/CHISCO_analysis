# train_sentence_type.py
#
# Same script as before, but with:
#  1) WeightedRandomSampler for TRAIN only
#  2) Class-weighted CrossEntropyLoss (kept)
#  3) Warmup = 10% of total steps
# Plus: balanced accuracy + confusion matrix in validation (kept)

import sys
import json
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

sys.path.append("./ds005170")

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


# ----------------- Data -----------------

def load_sentence_type_mapping(path: str):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {str(k): int(v) for k, v in m.items()}


def build_data(task, chisco_root, subject_codes):
    if task == "read":
        X, sentences, subj_ids = load_chisco_read_all_subjects(chisco_root, subject_codes)
    else:
        X, sentences, subj_ids = load_chisco_imagine_all_subjects(chisco_root, subject_codes)
    return X, sentences, subj_ids


def filter_by_sentence_type(X, sentences, subj_ids, mapping_path):
    sent2type = load_sentence_type_mapping(mapping_path)

    keep, labels = [], []
    for i, s in enumerate(sentences):
        if s in sent2type:
            keep.append(i)
            labels.append(sent2type[s])

    return (
        X[keep],
        [sentences[i] for i in keep],
        subj_ids[keep],
        torch.tensor(labels, dtype=torch.long),
    )


# ----------------- Splits -----------------

def make_train_val_indices(n, val_split, shuffle, seed):
    idx = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(idx)
    k = int(n * val_split)
    return idx[k:], idx[:k]


# ----------------- Model -----------------

class EEGToTextEmbedModel(nn.Module):
    def __init__(
        self,
        n_channels,
        n_subjects,
        n_classes,
        latent_dim,
        hidden_channels,
        final_hidden,
        n_blocks,
        pooling,
        use_eegnet,
        eegnet_temporal_kern,
        eegnet_F1,
        eegnet_D,
        use_spatial_attn,
        use_subject_layer,
        use_residual_blocks,
        raw_frontend_out_ch,
    ):
        super().__init__()
        self.pooling = pooling

        if use_eegnet:
            self.eeg_frontend = EEGFrontEnd(
                n_channels,
                eegnet_temporal_kern,
                eegnet_F1,
                eegnet_D,
            )
            fe_out = eegnet_F1 * eegnet_D
        else:
            self.eeg_frontend = nn.Conv1d(n_channels, raw_frontend_out_ch, 1)
            fe_out = raw_frontend_out_ch

        # IMPORTANT: use keyword args so pos_dim stays default=2
        self.brain = BrainModule(
            in_channels=fe_out,
            out_channels=latent_dim,
            n_subjects=n_subjects,
            hidden_channels=hidden_channels,
            final_hidden=final_hidden,
            n_blocks=n_blocks,
            spatial_out_channels=fe_out,
            use_spatial_attn=use_spatial_attn,
            use_subject_layer=use_subject_layer,
            use_residual_blocks=use_residual_blocks,
        )

        self.sensor_pos = nn.Parameter(torch.randn(fe_out, 2) * 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, n_classes),
        )

    def forward(self, x, subj_ids):
        x = self.eeg_frontend(x)
        z = self.brain(x, self.sensor_pos, subj_ids)
        z = z.mean(-1) if self.pooling == "mean" else z.max(-1)[0]
        return self.classifier(z)


# ----------------- Train / Eval -----------------

def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total, loss_sum = 0, 0.0
    for b in loader:
        eeg = b["eeg"].to(device)
        subj = b["subject_id"].to(device)
        y = b["target"].to(device)

        optimizer.zero_grad()
        logits = model(eeg, subj)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_sum += loss.item() * eeg.size(0)
        total += eeg.size(0)

    return loss_sum / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, loss_sum, correct = 0, 0.0, 0

    # Confusion matrix (rows=true, cols=pred): [[TN, FP],[FN, TP]]
    cm = [[0, 0], [0, 0]]

    for b in loader:
        eeg = b["eeg"].to(device)
        subj = b["subject_id"].to(device)
        y = b["target"].to(device)

        logits = model(eeg, subj)
        loss = criterion(logits, y)

        preds = logits.argmax(1)

        loss_sum += loss.item() * eeg.size(0)
        total += eeg.size(0)
        correct += (preds == y).sum().item()

        for yt, yp in zip(y.tolist(), preds.tolist()):
            cm[int(yt)][int(yp)] += 1

    acc = 100.0 * correct / max(1, total)

    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    denom0 = tn + fp
    denom1 = tp + fn
    recall0 = tn / denom0 if denom0 > 0 else 0.0
    recall1 = tp / denom1 if denom1 > 0 else 0.0
    bal_acc = 100.0 * (recall0 + recall1) / 2.0

    return loss_sum / max(1, total), acc, bal_acc, cm


# ----------------- Main -----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["read", "imagine"], default="read")
    p.add_argument("--sentence-type-json", default="./ds005170/json/question_statement_mapping.json")
    p.add_argument("--subject-codes", default="01,02,03,04,05")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    subject_codes = [s.strip() for s in args.subject_codes.split(",") if s.strip()]

    X, sentences, subj_ids = build_data(args.task, "./ds005170/derivatives/preprocessed_pkl", subject_codes)
    X, sentences, subj_ids, labels = filter_by_sentence_type(
        X, sentences, subj_ids, args.sentence_type_json
    )

    train_idx, val_idx = make_train_val_indices(len(labels), args.val_split, True, args.seed)

    # -------- Class-weighted loss (TRAIN ONLY stats) --------
    train_labels = labels[train_idx]
    class_counts = torch.bincount(train_labels)
    w0 = 1.0
    w1 = class_counts[0].float() / max(1.0, class_counts[1].float())
    class_weights = torch.tensor([w0, w1], device=device)

    print("Train class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # -------------------------------------------------------

    train_ds = ChiscoImagineDataset(
        X[train_idx], [sentences[i] for i in train_idx], subj_ids[train_idx], labels[train_idx]
    )
    val_ds = ChiscoImagineDataset(
        X[val_idx], [sentences[i] for i in val_idx], subj_ids[val_idx], labels[val_idx]
    )

    # -------- WeightedRandomSampler (TRAIN ONLY) --------
    # weight each sample inversely to its class frequency
    sample_weights = (1.0 / class_counts[train_labels]).double()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    # ---------------------------------------------------

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EEGToTextEmbedModel(
        n_channels=X.shape[1],
        n_subjects=len(subject_codes),
        n_classes=2,
        latent_dim=256,
        hidden_channels=256,
        final_hidden=256,
        n_blocks=5,
        pooling="mean",
        use_eegnet=True,
        eegnet_temporal_kern=64,
        eegnet_F1=16,
        eegnet_D=2,
        use_spatial_attn=True,
        use_subject_layer=True,
        use_residual_blocks=True,
        raw_frontend_out_ch=32,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)  # ---- Warmup = 10% ----

    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    for epoch in range(1, args.num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion)
        va_loss, va_acc, va_bal_acc, va_cm = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d} | train loss {tr_loss:.4f} | "
            f"val loss {va_loss:.4f} | val acc {va_acc:.2f}% | val bal-acc {va_bal_acc:.2f}%"
        )
        print("Val confusion matrix (rows=true, cols=pred) [[TN, FP],[FN, TP]]:")
        print(f"  [[{va_cm[0][0]}, {va_cm[0][1]}],")
        print(f"   [{va_cm[1][0]}, {va_cm[1][1]}]]")

    print("Training complete.")


if __name__ == "__main__":
    main()

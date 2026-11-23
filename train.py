"""Train an EEG -> sentence-embedding model with a symmetric contrastive (CLIP-style) loss.

Trials are split by sentence using a fixed split file, so no test sentence is
ever seen during training. The best checkpoint is chosen on validation Acc@10
and evaluated once on the test split at the end.

    python train.py --task read --seed 4
"""

import argparse
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from eeg2text.data import EEGTextDataset, load_all_trials
from eeg2text.experiment_logger import ExperimentLogger
from eeg2text.model import MODEL_DEFAULTS, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(eeg_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float):
    """Symmetric InfoNCE over the batch: trial i should match sentence i."""
    eeg_emb = nn.functional.normalize(eeg_emb, dim=-1)
    text_emb = nn.functional.normalize(text_emb, dim=-1)
    logits = eeg_emb @ text_emb.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss = (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels)) / 2
    return loss, logits


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, temperature, log_every_steps) -> float:
    model.train()
    total_loss, total = 0.0, 0

    for step, batch in enumerate(loader, start=1):
        eeg = batch["eeg"].to(device)
        loss, _ = contrastive_loss(model(eeg, batch["subject_id"].to(device)), batch["target"].to(device), temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(eeg)
        total += len(eeg)
        if log_every_steps > 0 and step % log_every_steps == 0:
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss {total_loss / total:.4f}")

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device, temperature) -> dict:
    """Contrastive loss and within-batch retrieval accuracy (Acc@1/5/10, in %)."""
    model.eval()
    sums = {"loss": 0.0, "acc1": 0.0, "acc5": 0.0, "acc10": 0.0}
    total = 0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        loss, logits = contrastive_loss(model(eeg, batch["subject_id"].to(device)), batch["target"].to(device), temperature)

        B = len(eeg)
        top = logits.topk(k=min(10, B), dim=1).indices
        correct = top.eq(torch.arange(B, device=device).unsqueeze(1))
        acc1 = correct[:, :1].any(dim=1).float().mean().item() * 100
        acc5 = correct[:, :5].any(dim=1).float().mean().item() * 100 if B >= 5 else acc1
        acc10 = correct[:, :10].any(dim=1).float().mean().item() * 100 if B >= 10 else acc5

        for key, value in (("loss", loss.item()), ("acc1", acc1), ("acc5", acc5), ("acc10", acc10)):
            sums[key] += value * B
        total += B

    return {key: value / total for key, value in sums.items()}


def save_training_plots(history: dict, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EEG→text contrastive loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eeg2text_loss_curve.png"))
    plt.close()

    plt.figure()
    for k in (1, 5, 10):
        plt.plot(epochs, history[f"val_acc{k}"], label=f"Val Acc@{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("EEG→text retrieval (within batch)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eeg2text_acc_curve.png"))
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = MODEL_DEFAULTS
    bool_flag = argparse.BooleanOptionalAction

    g = p.add_argument_group("data")
    g.add_argument("--chisco-root", default="./ds005170/derivatives/preprocessed_pkl")
    g.add_argument("--sent-emb-path", default="./ds005170/json/sentence_bert_embs.json")
    g.add_argument("--task", default="imagine", choices=["read", "imagine"])
    g.add_argument("--subject-codes", default="01,02,03,04,05", help="Comma-separated, e.g. 01,02,03")
    g.add_argument("--split-file", default="splits/train_val_test_split.json")
    g.add_argument("--train-data-fraction", type=float, default=1.0,
                   help="Train on a random fraction of the training trials (data-scaling experiments).")

    g = p.add_argument_group("training")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--num-epochs", type=int, default=50)
    g.add_argument("--learning-rate", type=float, default=2e-4)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--warmup-ratio", type=float, default=0.1)
    g.add_argument("--temperature", type=float, default=0.05)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--log-every-steps", type=int, default=50)
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    g = p.add_argument_group("augmentation")
    g.add_argument("--augment-train", action=bool_flag, default=True)
    g.add_argument("--noise-std", type=float, default=0.01)
    g.add_argument("--jitter-max", type=int, default=10, help="Max circular time shift, in samples.")
    g.add_argument("--channel-dropout-p", type=float, default=0.05)

    g = p.add_argument_group("model")
    g.add_argument("--arch", default=d["arch"], choices=["brainmodule", "chisco_baseline"])
    g.add_argument("--latent-dim", type=int, default=d["latent_dim"])
    g.add_argument("--hidden-channels", type=int, default=d["hidden_channels"])
    g.add_argument("--final-hidden", type=int, default=d["final_hidden"])
    g.add_argument("--n-blocks", type=int, default=d["n_blocks"])
    g.add_argument("--pooling", default=d["pooling"], choices=["mean", "max"])
    g.add_argument("--use-frontend", action=bool_flag, default=d["use_frontend"],
                   help="If disabled, raw EEG goes straight into the BrainModule (or head).")
    g.add_argument("--use-eegnet", action=bool_flag, default=d["use_eegnet"],
                   help="EEGNet-style front end; if disabled, a 1x1 convolution is used instead.")
    g.add_argument("--eegnet-temporal-kern", type=int, default=d["eegnet_temporal_kern"])
    g.add_argument("--eegnet-f1", type=int, default=d["eegnet_f1"])
    g.add_argument("--eegnet-d", type=int, default=d["eegnet_d"])
    g.add_argument("--raw-frontend-out-ch", type=int, default=d["raw_frontend_out_ch"],
                   help="Output channels of the 1x1 convolution front end.")
    g.add_argument("--use-brainmodule", action=bool_flag, default=d["use_brainmodule"],
                   help="If disabled, front-end features go straight to the projection head.")
    g.add_argument("--use-spatial-attn", action=bool_flag, default=d["use_spatial_attn"])
    g.add_argument("--use-subject-layer", action=bool_flag, default=d["use_subject_layer"])
    g.add_argument("--use-residual-blocks", action=bool_flag, default=d["use_residual_blocks"])

    g = p.add_argument_group("CHISCO baseline (--arch chisco_baseline)")
    g.add_argument("--baseline-layer", type=int, default=d["baseline_layer"], help="Transformer layers (0 = none).")
    g.add_argument("--baseline-pooling", default=d["baseline_pooling"], choices=["mean", "sums", "tops"])
    g.add_argument("--baseline-size1", type=int, default=d["baseline_size1"])
    g.add_argument("--baseline-size2", type=int, default=d["baseline_size2"])
    g.add_argument("--baseline-feel1", type=int, default=d["baseline_feel1"])
    g.add_argument("--baseline-feel2", type=int, default=d["baseline_feel2"])
    g.add_argument("--baseline-dropout1", type=float, default=d["baseline_dropout1"])
    g.add_argument("--baseline-dropout2", type=float, default=d["baseline_dropout2"])
    g.add_argument("--baseline-latent-dim", type=int, default=d["baseline_latent_dim"])

    g = p.add_argument_group("output")
    g.add_argument("--exp-root", default="experiments")
    g.add_argument("--experiment-name", default="eeg2text")
    g.add_argument("--no-plots", action="store_true")

    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    subject_codes = [s.strip() for s in args.subject_codes.split(",") if s.strip()]

    # --- data ---
    print(f"Loading CHISCO {args.task} trials...")
    X, sentences, subject_ids = load_all_trials(args.chisco_root, subject_codes, args.task)
    with open(args.sent_emb_path, "r", encoding="utf-8") as f:
        sent2emb = json.load(f)

    keep = [i for i, s in enumerate(sentences) if s in sent2emb]
    if len(keep) < len(sentences):
        print(f"Warning: skipping {len(sentences) - len(keep)} trials whose sentence has no embedding.")
    X, subject_ids = X[keep], subject_ids[keep]
    sentences = [sentences[i] for i in keep]
    targets = torch.tensor([sent2emb[s] for s in sentences], dtype=torch.float32)
    print(f"{len(X)} trials | {X.shape[1]} channels | {X.shape[2]} samples | text_dim={targets.shape[1]}")

    with open(args.split_file, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    def indices_for(split_name):
        split_sentences = {s.strip() for s in split_info[f"{split_name}_sentences"]}
        return [i for i, s in enumerate(sentences) if s in split_sentences]

    train_idx, val_idx, test_idx = indices_for("train"), indices_for("val"), indices_for("test")

    if args.train_data_fraction < 1.0:
        random.Random(args.seed).shuffle(train_idx)
        train_idx = train_idx[: max(1, int(len(train_idx) * args.train_data_fraction))]
    print(f"Trials | train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    split_info["train_data_fraction"] = args.train_data_fraction
    split_info["train_size_after_fraction"] = len(train_idx)

    def make_dataset(idx, augment):
        return EEGTextDataset(
            X[idx], [sentences[i] for i in idx], subject_ids[idx], targets[idx],
            augment=augment, noise_std=args.noise_std, jitter_max=args.jitter_max,
            channel_dropout_p=args.channel_dropout_p,
        )

    def make_loader(idx, augment=False, shuffle=False):
        return DataLoader(make_dataset(idx, augment), batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader(train_idx, augment=args.augment_train, shuffle=True)
    val_loader = make_loader(val_idx)
    test_loader = make_loader(test_idx)

    # --- model ---
    config = vars(args).copy()
    config["subject_codes"] = subject_codes
    model = build_model(config, X.shape[1], len(subject_codes), targets.shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * args.warmup_ratio), total_steps)

    # Runs are saved under <exp-root>/<experiment-name>/<task>/<split-mode>/<run_id>/
    logger = ExperimentLogger(
        root=args.exp_root,
        task_name=os.path.join(args.experiment_name, args.task, split_info["split_mode"]),
        config=config,
        split_info=split_info,
        metric_name="val_acc10",
    )
    print("Logging to:", logger.run_dir)

    # --- training ---
    history = {"train_loss": [], "val_loss": [], "val_acc1": [], "val_acc5": [], "val_acc10": []}

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.num_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch,
                                     args.temperature, args.log_every_steps)
        val = evaluate(model, val_loader, device, args.temperature)
        print(f"Train | Loss {train_loss:.4f}")
        print(f"Val   | Loss {val['loss']:.4f} | Acc@1 {val['acc1']:.2f}% | "
              f"Acc@5 {val['acc5']:.2f}% | Acc@10 {val['acc10']:.2f}%")

        history["train_loss"].append(train_loss)
        for key in ("loss", "acc1", "acc5", "acc10"):
            history[f"val_{key}"].append(val[key])

        logger.log_epoch(epoch, {"loss": train_loss}, val)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "split_info": split_info,
            "train": {"loss": train_loss},
            "val": val,
        }
        logger.save_last(checkpoint)
        if logger.maybe_save_best(epoch, val["acc10"], checkpoint):
            print(f"New best val Acc@10 = {val['acc10']:.2f}% (saved best_checkpoint.pt)")

    # --- test, using the best-on-validation checkpoint ---
    best = torch.load(logger.best_checkpoint_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, test_loader, device, args.temperature)
    print(f"\nTest (best epoch {logger.best_epoch}) | Loss {test['loss']:.4f} | Acc@1 {test['acc1']:.2f}% | "
          f"Acc@5 {test['acc5']:.2f}% | Acc@10 {test['acc10']:.2f}%")

    if not args.no_plots:
        save_training_plots(history, os.path.join(logger.run_dir, "plots"))

    print(f"\nBest val Acc@10: {logger.best_value:.2f}%")
    print("Run directory:", logger.run_dir)


if __name__ == "__main__":
    main()

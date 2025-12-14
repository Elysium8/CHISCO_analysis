# eval_global_retrieval.py
#
# Evaluate EEG→text embeddings by retrieving the correct sentence
# from the *entire* text embedding vocabulary (global retrieval).
#
# Uses the best_checkpoint saved by train_eeg2text_model.py

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

# Make ds005170 importable
sys.path.append("./ds005170")

from data_loader import (
    load_chisco_imagine_all_subjects,
    ChiscoImagineDataset,
)
from train_eeg_to_text_embed import EEGToTextEmbedModel, Config


def load_sentence_embeddings(path):
    with open(path, "r", encoding="utf-8") as f:
        sent2emb = json.load(f)
    return sent2emb


def build_text_embedding_matrix(sent2emb):
    """
    Creates:
        matrix: (N_sent, D)
        id_to_sentence: list of sentences aligned to matrix rows
        sentence_to_id: reverse lookup
    """
    sentences = sorted(sent2emb.keys())
    id_to_sentence = sentences
    sentence_to_id = {s: i for i, s in enumerate(sentences)}

    matrix = np.stack([np.array(sent2emb[s], dtype=np.float32) for s in sentences], axis=0)
    matrix = torch.tensor(matrix)  # (N_sent, D)

    # L2 normalize the entire matrix
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    return matrix, id_to_sentence, sentence_to_id


def load_best_model(checkpoint_path, cfg, n_channels, n_subjects, text_dim, device):
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = EEGToTextEmbedModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        text_dim=text_dim,
        latent_dim=cfg.LATENT_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        final_hidden=cfg.FINAL_HIDDEN,
        n_blocks=cfg.N_BLOCKS,
        pooling=cfg.POOLING,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_global_retrieval(model, X, sentences, subj_ids, text_matrix, sentence_to_id, device):
    """
    Compute global retrieval:
    For each EEG trial:
        1. Compute EEG embedding
        2. Compute cosine similarity to ALL sentence embeddings
        3. Check top-k accuracy
    """

    N = X.shape[0]
    text_matrix = text_matrix.to(device)  # (N_sent, D)
    BERT_dim = text_matrix.shape[1]

    eeg_dataset = ChiscoImagineDataset(
        eeg=X,
        sentences=sentences,
        subject_ids=subj_ids,
        targets=None,
        augment=False,
    )

    loader = DataLoader(eeg_dataset, batch_size=32, shuffle=False)

    top1 = 0
    top5 = 0
    top10 = 0
    top50 = 0

    total = 0

    for batch in loader:
        eeg = batch["eeg"].to(device)          # (B, C, T)
        subj = batch["subject_id"].to(device)  # (B,)
        batch_sentences = batch["sentence"]

        # 1) Compute EEG embedding
        eeg_emb = model(eeg, subj)             # (B, D)
        eeg_emb = torch.nn.functional.normalize(eeg_emb, dim=1)

        # 2) Compute cosine similarity against entire text embedding matrix
        #    sim: (B, N_sent)
        sim = eeg_emb @ text_matrix.T

        # 3) For each trial, check ranking
        values, indices = sim.topk(k=50, dim=1)   # get top-50

        for i, sent in enumerate(batch_sentences):
            total += 1
            true_id = sentence_to_id[sent]
            top_candidates = indices[i].tolist()

            if true_id == top_candidates[0]:
                top1 += 1
            if true_id in top_candidates[:5]:
                top5 += 1
            if true_id in top_candidates[:10]:
                top10 += 1
            if true_id in top_candidates[:50]:
                top50 += 1

    return {
        "top1": top1 / total * 100,
        "top5": top5 / total * 100,
        "top10": top10 / total * 100,
        "top50": top50 / total * 100,
        "total": total
    }


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ---------------------------------------------------------
    # 1. Load all validation EEG trials (same split as training)
    # ---------------------------------------------------------
    print("Loading CHISCO data...")
    X, sentences, subj_ids = load_chisco_imagine_all_subjects(
        root=cfg.CHISCO_ROOT,
        subject_codes=cfg.SUBJECT_CODES,
    )
    n_samples, n_channels, _ = X.shape
    n_subjects = len(cfg.SUBJECT_CODES)

    # Use the same split method as in training
    from train_eeg_to_text_embed import make_train_val_indices
    train_idx, val_idx = make_train_val_indices(
        n_samples=n_samples,
        val_split=cfg.VAL_SPLIT,
        shuffle=cfg.SHUFFLE_DATA,
        seed=cfg.RANDOM_SEED,
    )

    X_val = X[val_idx]
    sentences_val = [sentences[idx] for idx in val_idx]
    subj_ids_val = subj_ids[val_idx]

    print(f"Validation trials: {len(val_idx)}")

    # ---------------------------------------------------------
    # 2. Load text embeddings
    # ---------------------------------------------------------
    print("Loading text embeddings...")
    sent2emb = load_sentence_embeddings(cfg.SENT_EMB_PATH)
    text_matrix, id_to_sentence, sentence_to_id = build_text_embedding_matrix(sent2emb)
    text_dim = text_matrix.shape[1]

    print(f"Total sentences in embedding vocabulary: {len(id_to_sentence)}")

    # ---------------------------------------------------------
    # 3. Load the best saved model
    # ---------------------------------------------------------
    ckpt_path = "checkpoints/best_eeg2text_model.pt"
    model = load_best_model(
        ckpt_path,
        cfg,
        n_channels=n_channels,
        n_subjects=n_subjects,
        text_dim=text_dim,
        device=device
    )

    # ---------------------------------------------------------
    # 4. Run global retrieval evaluation!
    # ---------------------------------------------------------
    print("\nRunning global retrieval evaluation...")
    results = evaluate_global_retrieval(
        model,
        X_val,
        sentences_val,
        subj_ids_val,
        text_matrix,
        sentence_to_id,
        device,
    )

    print("\n=== Global Retrieval Accuracy ===")
    print(f"Top-1  : {results['top1']:.2f}%")
    print(f"Top-5  : {results['top5']:.2f}%")
    print(f"Top-10 : {results['top10']:.2f}%")
    print(f"Top-50 : {results['top50']:.2f}%")
    print(f"(Out of {results['total']} validation trials)")

    print("\nDone.")


if __name__ == "__main__":
    main()

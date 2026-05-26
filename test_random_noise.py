# eval_random_noise_baseline.py
#
# Minimal random-noise baseline for global retrieval.
# - Always uses random EEG noise shaped like the real EEG for the task (read vs imagine).
# - Uses your existing split.json + checkpoint model + sentence embedding vocab.

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Make ds005170 importable
import sys
sys.path.append("./ds005170")
from data_loader import load_read_run, load_imagine_run  # type: ignore

try:
    from updated_train_eeg_to_text_embed import EEGToTextEmbedModel
except Exception:
    from train_eeg_to_text_embed import EEGToTextEmbedModel


def infer_ct(chisco_root: str, subject_codes, task: str):
    run_loader = load_read_run if task == "read" else load_imagine_run
    for sub in subject_codes:
        for run_idx in range(1, 46):
            trials = run_loader(chisco_root, sub, run_idx)
            if not trials:
                continue
            for tr in trials:
                if "input_features" not in tr:
                    continue
                arr = tr["input_features"]          # (1, 125, T_raw)
                eeg = arr[0, :122, :]               # (122, T)  (same as your eval)  :contentReference[oaicite:1]{index=1}
                return int(eeg.shape[0]), int(eeg.shape[1])
    raise RuntimeError(f"Could not infer EEG shape for task={task} from {chisco_root}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_matrix(sent_emb_path: str):
    sent2emb = load_json(sent_emb_path)
    sents = sorted(sent2emb.keys())
    s2i = {s: i for i, s in enumerate(sents)}
    mat = np.stack([np.asarray(sent2emb[s], np.float32) for s in sents], axis=0)
    mat = torch.tensor(mat)
    mat = torch.nn.functional.normalize(mat, dim=1)
    return mat, s2i


def build_trial_index(chisco_root: str, subject_codes, task: str):
    run_loader = load_read_run if task == "read" else load_imagine_run
    refs = []
    for subj_idx, sub in enumerate(subject_codes):
        for run_idx in range(1, 46):
            trials = run_loader(chisco_root, sub, run_idx)
            if not trials:
                continue
            for t_i, tr in enumerate(trials):
                refs.append((subj_idx, str(tr["text"]).strip()))
    return refs  # list of (subject_index, sentence)


class RandomNoiseDataset(Dataset):
    def __init__(self, refs, c: int, t: int, seed: int = 0):
        self.refs = refs
        self.c = c
        self.t = t
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.refs)

    def __getitem__(self, idx):
        subj_idx, sent = self.refs[idx]
        eeg = self.rng.standard_normal((self.c, self.t), dtype=np.float32)
        return {
            "eeg": torch.from_numpy(eeg),
            "subject_id": torch.tensor(subj_idx, dtype=torch.long),
            "sentence": sent,
        }


@torch.no_grad()
def eval_retrieval(model, ds, text_mat, s2i, device, batch_size=32, topk=50):
    model.eval()
    text_mat = text_mat.to(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    hits = {1: 0, 5: 0, 10: 0, 50: 0}
    total = 0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_id"].to(device)
        sents = batch["sentence"]

        z = torch.nn.functional.normalize(model(eeg, subj), dim=1)
        sims = z @ text_mat.T
        k = min(topk, sims.shape[1])
        _, idxs = sims.topk(k=k, dim=1)

        for i, s in enumerate(sents):
            total += 1
            true = s2i.get(s, None)
            if true is None:
                continue
            cand = idxs[i].tolist()
            if true == cand[0]: hits[1] += 1
            if true in cand[:5]: hits[5] += 1
            if true in cand[:10]: hits[10] += 1
            if true in cand[:50]: hits[50] += 1

    return {
        "top1": hits[1] / total * 100.0,
        "top5": hits[5] / total * 100.0,
        "top10": hits[10] / total * 100.0,
        "top50": hits[50] / total * 100.0,
        "total": float(total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--noise-seed", type=int, default=0)
    args = ap.parse_args()

    exp_dir = args.exp_dir
    cfg = load_json(os.path.join(exp_dir, "config.json"))
    split_info = load_json(os.path.join(exp_dir, "split.json"))
    ckpt_path = os.path.join(exp_dir, "best_checkpoint.pt")

    chisco_root = cfg.get("chisco_root") or "./ds005170/derivatives/preprocessed_pkl"
    sent_emb_path = cfg.get("sent_emb_path") or "./ds005170/json/sentence_bert_embs.json"
    task = (cfg.get("task") or "read").lower()

    subject_codes = cfg.get("subject_codes")
    if isinstance(subject_codes, str):
        subject_codes = [s.strip() for s in subject_codes.split(",") if s.strip()]
    elif not isinstance(subject_codes, list):
        subject_codes = [f"{i:02d}" for i in range(1, 6)]

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Trial index + exact val split indices (reused directly)
    refs = build_trial_index(chisco_root, subject_codes, task)
    val_idx = split_info["val_idx"]

    # Infer (C,T) from real data for this task, then generate noise with same shape
    c, t = infer_ct(chisco_root, subject_codes, task)
    base = RandomNoiseDataset(refs, c=c, t=t, seed=args.noise_seed)
    val_ds = Subset(base, val_idx)

    # Text vocab matrix
    text_mat, s2i = build_text_matrix(sent_emb_path)
    text_dim = int(text_mat.shape[1])

    # Model from checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    c2 = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    def _get(k, d): return c2.get(k, cfg.get(k, d))

    model = EEGToTextEmbedModel(
        n_channels=c,
        n_subjects=len(subject_codes),
        text_dim=text_dim,
        latent_dim=int(_get("latent_dim", 256)),
        hidden_channels=int(_get("hidden_channels", 256)),
        final_hidden=int(_get("final_hidden", 256)),
        n_blocks=int(_get("n_blocks", 5)),
        pooling=_get("pooling", "mean"),
        use_eegnet=bool(_get("use_eegnet", True)),
        eegnet_temporal_kern=int(_get("eegnet_temporal_kern", 64)),
        eegnet_F1=int(_get("eegnet_f1", 16)),
        eegnet_D=int(_get("eegnet_d", 2)),
        use_spatial_attn=bool(_get("use_spatial_attn", True)),
        use_subject_layer=bool(_get("use_subject_layer", True)),
        use_residual_blocks=bool(_get("use_residual_blocks", True)),
        raw_frontend_out_ch=int(_get("raw_frontend_out_ch", 32)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    results = eval_retrieval(model, val_ds, text_mat, s2i, device, batch_size=args.batch_size)

    out_path = os.path.join(exp_dir, "global_retrieval_random_noise.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"task": task, "noise_seed": args.noise_seed, "eeg_shape": [c, t], **results},
            f,
            indent=2,
        )

    print(f"Task: {task} | EEG shape (C,T)=({c},{t}) | seed={args.noise_seed}")
    print(f"Top-1  : {results['top1']:.2f}%")
    print(f"Top-5  : {results['top5']:.2f}%")
    print(f"Top-10 : {results['top10']:.2f}%")
    print(f"Top-50 : {results['top50']:.2f}%")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

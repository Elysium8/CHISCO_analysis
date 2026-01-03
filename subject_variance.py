# subject_variance.py
#
# Covariance/correlation (temporally invariant) comparison across subjects for CHISCO READ vs IMAGINE
# with FIX 1: enforce a common time window length T_common before covariance estimation
#
# Outputs (in --outdir):
#  - read_pca_shared_axes.png / imagine_pca_shared_axes.png  (same PCA basis + same axis limits)
#  - read_report.json / imagine_report.json
#  - read_centroid_distance_matrix.csv / imagine_centroid_distance_matrix.csv
#  - summary.json
#
# Notes:
#  - This script keeps your ChiscoStreamingDataset happy by loading sentence embeddings, but it does not use targets.
#  - The key fix is cropping (or selecting) the same number of time samples for all segments before covariance.
#
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.covariance import LedoitWolf
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score
from scipy.linalg import logm
from itertools import combinations

import matplotlib.pyplot as plt

from data_loader import ChiscoStreamingDataset


# ----------------------------
# Load sentence embeddings (to keep dataset happy)
# ----------------------------

def load_sentence_embeddings_json(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}


# ----------------------------
# FIX 1: enforce common window length T_common
# ----------------------------

def pick_window(X_ct: np.ndarray, T_common: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """
    X_ct: (C, T)
    Returns (C, T_common) using a deterministic rule (start/center) or seeded random.
    If T < T_common, we pad with zeros (shouldn't happen if T_common is min across conditions).
    """
    T = X_ct.shape[1]
    if T == T_common:
        return X_ct

    if T > T_common:
        if mode == "start":
            s = 0
        elif mode == "center":
            s = (T - T_common) // 2
        elif mode == "random":
            s = int(rng.integers(0, T - T_common + 1))
        else:
            raise ValueError(f"Unknown window_mode: {mode}")
        return X_ct[:, s:s + T_common]

    # T < T_common: pad (rare/unexpected)
    out = np.zeros((X_ct.shape[0], T_common), dtype=X_ct.dtype)
    out[:, :T] = X_ct
    return out


# ----------------------------
# Segment -> correlation -> log-Euclidean vector
# ----------------------------

def segment_to_corr(
    X_ct: np.ndarray,
    shrinkage: bool = True,
    eps: float = 1e-8,
    T_common: int | None = None,
    window_mode: str = "center",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    X_ct: (C, T) segment
    Returns: (C, C) correlation matrix (SPD-ish)
    """
    if T_common is not None:
        if rng is None:
            rng = np.random.default_rng(0)
        X_ct = pick_window(X_ct, T_common=T_common, mode=window_mode, rng=rng)

    X = X_ct.astype(np.float64, copy=False)
    X = X - X.mean(axis=1, keepdims=True)

    if shrinkage:
        cov = LedoitWolf().fit(X.T).covariance_
    else:
        cov = (X @ X.T) / max(1, X.shape[1] - 1)

    d = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / (d[:, None] * d[None, :])
    corr = 0.5 * (corr + corr.T)
    corr += eps * np.eye(corr.shape[0])
    return corr


def logeuclid_vec(A: np.ndarray) -> np.ndarray:
    L = logm(A).real
    L = 0.5 * (L + L.T)
    iu = np.triu_indices(L.shape[0])
    return L[iu].astype(np.float32, copy=False)


def batch_embed(
    eeg_batch_bct: torch.Tensor,
    shrinkage: bool = True,
    T_common: int | None = None,
    window_mode: str = "center",
    seed: int = 0,
) -> np.ndarray:
    """
    eeg_batch_bct: (B, C, T)
    Returns: (B, D) where D = C(C+1)/2
    """
    X = eeg_batch_bct.detach().cpu().numpy()
    rng = np.random.default_rng(seed)

    out = []
    for i in range(X.shape[0]):
        corr = segment_to_corr(
            X[i],
            shrinkage=shrinkage,
            T_common=T_common,
            window_mode=window_mode,
            rng=rng,
        )
        out.append(logeuclid_vec(corr))
    return np.stack(out, axis=0)


# ----------------------------
# Running mean per subject
# ----------------------------

class RunningMean:
    def __init__(self):
        self.n = 0
        self.mean = None

    def update(self, x: np.ndarray):
        if self.mean is None:
            self.mean = x.astype(np.float64, copy=True)
            self.n = 1
            return
        self.n += 1
        self.mean += (x - self.mean) / self.n

    def value(self):
        return self.mean


# ----------------------------
# Pass 1: fit ONE shared PCA across modes + compute per-mode subject centroids (in full embedding space)
# ----------------------------

def pass1_fit_shared_pca_and_centroids(
    loaders_by_mode: dict,
    n_subjects: int,
    ipca: IncrementalPCA,
    shrinkage: bool,
    T_common: int | None,
    window_mode: str,
    seed: int,
    log_every_batches: int = 50,
):
    subj_means_by_mode = {mode: [RunningMean() for _ in range(n_subjects)] for mode in loaders_by_mode.keys()}
    total_seen_by_mode = {mode: 0 for mode in loaders_by_mode.keys()}

    for mode, loader in loaders_by_mode.items():
        print(f"\n[Pass 1] Fitting shared PCA on mode={mode} ...", flush=True)

        # mode-specific seed offset for reproducible random windowing
        mode_seed = seed + (0 if mode == "READ" else 10_000)

        for batch_idx, batch in enumerate(loader):
            eeg = batch["eeg"]
            subj = batch["subject_id"].detach().cpu().numpy().astype(int)

            emb = batch_embed(
                eeg,
                shrinkage=shrinkage,
                T_common=T_common,
                window_mode=window_mode,
                seed=mode_seed + batch_idx,  # vary per batch deterministically
            )

            ipca.partial_fit(emb)

            for i in range(emb.shape[0]):
                subj_means_by_mode[mode][subj[i]].update(emb[i])

            total_seen_by_mode[mode] += emb.shape[0]

            if log_every_batches and (batch_idx % log_every_batches == 0):
                print(f"[Pass 1][{mode}] processed ~{batch_idx * loader.batch_size} segments", flush=True)

        print(f"[Pass 1][{mode}] done. total segments: {total_seen_by_mode[mode]}", flush=True)

    return subj_means_by_mode


def centroid_distance_matrix(subj_means: list[RunningMean]) -> np.ndarray:
    centroids = [m.value() for m in subj_means]
    present = [i for i, c in enumerate(centroids) if c is not None]

    M = np.full((len(centroids), len(centroids)), np.nan, dtype=np.float64)
    for i in present:
        M[i, i] = 0.0
    for a, b in combinations(present, 2):
        d = float(np.linalg.norm(centroids[a] - centroids[b]))
        M[a, b] = d
        M[b, a] = d
    return M


def avg_between_subject_distance_from_matrix(M: np.ndarray) -> float:
    vals = M[np.isfinite(M) & (M > 0)]
    return float(np.mean(vals)) if vals.size else float("nan")


# ----------------------------
# Pass 2: collect points for plotting + sampled full-D points for stats
# ----------------------------

def pass2_collect_points(
    loader,
    ipca: IncrementalPCA,
    shrinkage: bool,
    T_common: int | None,
    window_mode: str,
    seed: int,
    max_points_per_subject: int = 2000,
    max_fullD_per_subject: int = 800,
):
    rng = np.random.default_rng(seed)

    points2d = {}     # subj -> list of (2,)
    pointsD = {}      # subj -> list of (D,)
    counts = {}       # subj -> seen

    for batch_idx, batch in enumerate(loader):
        eeg = batch["eeg"]
        subj = batch["subject_id"].detach().cpu().numpy().astype(int)

        emb = batch_embed(
            eeg,
            shrinkage=shrinkage,
            T_common=T_common,
            window_mode=window_mode,
            seed=seed + batch_idx,
        )
        z2 = ipca.transform(emb)[:, :2].astype(np.float32)

        for i in range(z2.shape[0]):
            s = int(subj[i])
            counts[s] = counts.get(s, 0) + 1

            points2d.setdefault(s, [])
            pointsD.setdefault(s, [])

            # reservoir sampling for 2D
            if len(points2d[s]) < max_points_per_subject:
                points2d[s].append(z2[i])
            else:
                j = int(rng.integers(0, counts[s]))
                if j < max_points_per_subject:
                    points2d[s][j] = z2[i]

            # reservoir sampling for full-D (smaller)
            if len(pointsD[s]) < max_fullD_per_subject:
                pointsD[s].append(emb[i])
            else:
                j = int(rng.integers(0, counts[s]))
                if j < max_fullD_per_subject:
                    pointsD[s][j] = emb[i]

    all_xy, all_s = [], []
    all_D, all_sD = [], []

    for s, pts in points2d.items():
        pts = np.stack(pts, axis=0)
        all_xy.append(pts)
        all_s.append(np.full((pts.shape[0],), s, dtype=int))

    for s, pts in pointsD.items():
        pts = np.stack(pts, axis=0)
        all_D.append(pts)
        all_sD.append(np.full((pts.shape[0],), s, dtype=int))

    if not all_xy:
        raise RuntimeError("No points collected. Dataset produced no batches?")

    xy = np.concatenate(all_xy, axis=0)
    sid = np.concatenate(all_s, axis=0)
    XD = np.concatenate(all_D, axis=0)
    sidD = np.concatenate(all_sD, axis=0)
    return xy, sid, XD, sidD


# ----------------------------
# Plotting (fixed axis limits across modes)
# ----------------------------

def plot_2d_fixed_axes(xy, subj_ids, title, outpath, xlim, ylim):
    plt.figure()
    for s in np.unique(subj_ids):
        m = subj_ids == s
        plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.7, label=f"subj {s}")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved:", outpath)


# ----------------------------
# Stats + saving
# ----------------------------

def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def save_matrix_csv(M: np.ndarray, outpath: str, subject_codes: list[str]):
    header = ",".join([""] + [f"subj_{i}_{code}" for i, code in enumerate(subject_codes)])
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(M.shape[0]):
            row = [f"subj_{i}_{subject_codes[i]}"]
            for j in range(M.shape[1]):
                v = M[i, j]
                row.append("" if not np.isfinite(v) else f"{v:.6f}")
            f.write(",".join(row) + "\n")
    print("Saved:", outpath)


def compute_mode_report(
    mode_name: str,
    subject_codes: list[str],
    subj_means: list[RunningMean],
    xy: np.ndarray,
    sid: np.ndarray,
    XD: np.ndarray,
    sidD: np.ndarray,
    pca: IncrementalPCA,
):
    M = centroid_distance_matrix(subj_means)
    avg_between = avg_between_subject_distance_from_matrix(M)

    centroids = [m.value() for m in subj_means]
    within = {}
    for s in range(len(subject_codes)):
        c = centroids[s]
        if c is None:
            within[s] = float("nan")
            continue
        Xs = XD[sidD == s]
        if Xs.shape[0] == 0:
            within[s] = float("nan")
            continue
        within[s] = float(np.mean(np.linalg.norm(Xs - c[None, :], axis=1)))

    within_vals = np.array([within[s] for s in range(len(subject_codes))], dtype=np.float64)
    within_mean = float(np.nanmean(within_vals))
    within_std = float(np.nanstd(within_vals))

    bw_ratio = float(avg_between / within_mean) if np.isfinite(avg_between) and within_mean > 0 else float("nan")

    sil_2d = safe_silhouette(xy, sid)
    sil_D = safe_silhouette(XD, sidD)

    evr = getattr(pca, "explained_variance_ratio_", None)
    evr_list = evr.tolist() if evr is not None else []

    report = {
        "mode": mode_name,
        "n_subjects": len(subject_codes),
        "subjects": [{"subject_index": i, "subject_code": subject_codes[i]} for i in range(len(subject_codes))],
        "avg_between_subject_centroid_distance": avg_between,
        "within_subject_dispersion_mean": within_mean,
        "within_subject_dispersion_std": within_std,
        "between_over_within_ratio": bw_ratio,
        "within_subject_dispersion_by_subject_index": within,
        "silhouette_score_2d_pca": sil_2d,
        "silhouette_score_full_embedding_sample": sil_D,
        "pca_explained_variance_ratio": evr_list,
        "pca_explained_variance_ratio_pc1": evr_list[0] if len(evr_list) > 0 else None,
        "pca_explained_variance_ratio_pc2": evr_list[1] if len(evr_list) > 1 else None,
    }
    return report, M


# ----------------------------
# Dataset/loader helpers
# ----------------------------

def make_dataset_and_loader(chisco_root, subject_codes, task, sent2emb, batch_size, num_workers):
    ds = ChiscoStreamingDataset(
        root=chisco_root,
        task=task,
        subject_codes=subject_codes,
        sent2emb=sent2emb,
        cache_runs=2,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds, loader


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chisco-root", type=str, default="./ds005170/derivatives/preprocessed_pkl")
    ap.add_argument("--subject-codes", type=str, default="01,02,03,04,05")
    ap.add_argument("--sent-emb-path", type=str, default="ds005170/json/sentence_bert_embs.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pca-components", type=int, default=10)
    ap.add_argument("--no-shrinkage", action="store_true")
    ap.add_argument("--outdir", type=str, default="covcorr_report")
    ap.add_argument("--max-plot-points-per-subject", type=int, default=2000)
    ap.add_argument("--max-fulld-per-subject", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)

    # FIX 1 args
    ap.add_argument("--t-common", type=int, default=1651,
                    help="Force all segments to this many time samples before covariance. Use 0 to disable.")
    ap.add_argument("--window-mode", type=str, default="center",
                    choices=["start", "center", "random"],
                    help="How to select the T_common window from longer segments.")
    args = ap.parse_args()

    shrinkage = (not args.no_shrinkage)
    T_common = args.t_common if args.t_common and args.t_common > 0 else None

    subject_codes = [s.strip() for s in args.subject_codes.split(",") if s.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    sent2emb = load_sentence_embeddings_json(args.sent_emb_path)

    # Pass 1 loaders
    _, loader_read = make_dataset_and_loader(args.chisco_root, subject_codes, "read", sent2emb,
                                            args.batch_size, args.num_workers)
    _, loader_imag = make_dataset_and_loader(args.chisco_root, subject_codes, "imagine", sent2emb,
                                            args.batch_size, args.num_workers)

    ipca = IncrementalPCA(n_components=args.pca_components)

    subj_means_by_mode = pass1_fit_shared_pca_and_centroids(
        loaders_by_mode={"READ": loader_read, "IMAGINE": loader_imag},
        n_subjects=len(subject_codes),
        ipca=ipca,
        shrinkage=shrinkage,
        T_common=T_common,
        window_mode=args.window_mode,
        seed=args.seed,
        log_every_batches=50,
    )

    # Pass 2 loaders (fresh)
    _, loader_read2 = make_dataset_and_loader(args.chisco_root, subject_codes, "read", sent2emb,
                                             args.batch_size, args.num_workers)
    _, loader_imag2 = make_dataset_and_loader(args.chisco_root, subject_codes, "imagine", sent2emb,
                                             args.batch_size, args.num_workers)

    xy_read, sid_read, XD_read, sidD_read = pass2_collect_points(
        loader_read2, ipca,
        shrinkage=shrinkage,
        T_common=T_common,
        window_mode=args.window_mode,
        seed=args.seed + 1000,
        max_points_per_subject=args.max_plot_points_per_subject,
        max_fullD_per_subject=args.max_fulld_per_subject,
    )
    xy_imag, sid_imag, XD_imag, sidD_imag = pass2_collect_points(
        loader_imag2, ipca,
        shrinkage=shrinkage,
        T_common=T_common,
        window_mode=args.window_mode,
        seed=args.seed + 2000,
        max_points_per_subject=args.max_plot_points_per_subject,
        max_fullD_per_subject=args.max_fulld_per_subject,
    )

    # Shared axes across modes
    all_xy = np.concatenate([xy_read, xy_imag], axis=0)
    x_pad = 0.05 * (all_xy[:, 0].max() - all_xy[:, 0].min() + 1e-6)
    y_pad = 0.05 * (all_xy[:, 1].max() - all_xy[:, 1].min() + 1e-6)
    xlim = (float(all_xy[:, 0].min() - x_pad), float(all_xy[:, 0].max() + x_pad))
    ylim = (float(all_xy[:, 1].min() - y_pad), float(all_xy[:, 1].max() + y_pad))

    # Reports
    report_read, M_read = compute_mode_report(
        "READ", subject_codes, subj_means_by_mode["READ"],
        xy_read, sid_read, XD_read, sidD_read, ipca
    )
    report_imag, M_imag = compute_mode_report(
        "IMAGINE", subject_codes, subj_means_by_mode["IMAGINE"],
        xy_imag, sid_imag, XD_imag, sidD_imag, ipca
    )

    # Save plots
    plot_2d_fixed_axes(
        xy_read, sid_read,
        f"CHISCO READ: log-corr embeddings (shared PCA, T_common={T_common}, window={args.window_mode})",
        os.path.join(args.outdir, "read_pca_shared_axes.png"),
        xlim, ylim
    )
    plot_2d_fixed_axes(
        xy_imag, sid_imag,
        f"CHISCO IMAGINE: log-corr embeddings (shared PCA, T_common={T_common}, window={args.window_mode})",
        os.path.join(args.outdir, "imagine_pca_shared_axes.png"),
        xlim, ylim
    )

    # Save matrices
    save_matrix_csv(M_read, os.path.join(args.outdir, "read_centroid_distance_matrix.csv"), subject_codes)
    save_matrix_csv(M_imag, os.path.join(args.outdir, "imagine_centroid_distance_matrix.csv"), subject_codes)

    # Save reports
    with open(os.path.join(args.outdir, "read_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_read, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(args.outdir, "read_report.json"))

    with open(os.path.join(args.outdir, "imagine_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_imag, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(args.outdir, "imagine_report.json"))

    # Summary
    summary = {
        "subjects": [{"subject_index": i, "subject_code": subject_codes[i]} for i in range(len(subject_codes))],
        "shared_pca_components": args.pca_components,
        "t_common": T_common,
        "window_mode": args.window_mode,
        "shrinkage": shrinkage,
        "pca_explained_variance_ratio": getattr(ipca, "explained_variance_ratio_", None).tolist()
        if getattr(ipca, "explained_variance_ratio_", None) is not None else [],
        "READ": {
            "avg_between_subject_centroid_distance": report_read["avg_between_subject_centroid_distance"],
            "within_subject_dispersion_mean": report_read["within_subject_dispersion_mean"],
            "within_subject_dispersion_std": report_read["within_subject_dispersion_std"],
            "between_over_within_ratio": report_read["between_over_within_ratio"],
            "silhouette_score_2d_pca": report_read["silhouette_score_2d_pca"],
            "silhouette_score_full_embedding_sample": report_read["silhouette_score_full_embedding_sample"],
        },
        "IMAGINE": {
            "avg_between_subject_centroid_distance": report_imag["avg_between_subject_centroid_distance"],
            "within_subject_dispersion_mean": report_imag["within_subject_dispersion_mean"],
            "within_subject_dispersion_std": report_imag["within_subject_dispersion_std"],
            "between_over_within_ratio": report_imag["between_over_within_ratio"],
            "silhouette_score_2d_pca": report_imag["silhouette_score_2d_pca"],
            "silhouette_score_full_embedding_sample": report_imag["silhouette_score_full_embedding_sample"],
        },
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(args.outdir, "summary.json"))

    print("\n=== Quick summary ===")
    print(f"READ    avg between-subject centroid distance: {summary['READ']['avg_between_subject_centroid_distance']:.4f}")
    print(f"IMAGINE avg between-subject centroid distance: {summary['IMAGINE']['avg_between_subject_centroid_distance']:.4f}")
    print(f"READ    silhouette (2D PCA): {summary['READ']['silhouette_score_2d_pca']:.4f}")
    print(f"IMAGINE silhouette (2D PCA): {summary['IMAGINE']['silhouette_score_2d_pca']:.4f}")


if __name__ == "__main__":
    main()

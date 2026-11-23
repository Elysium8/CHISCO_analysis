"""How strongly do trials cluster by subject, for reading vs imagined speech?

Each trial is summarised by its channel correlation matrix (Ledoit-Wolf
shrinkage), mapped to a vector with the log-Euclidean transform. With one PCA
basis shared by both tasks we then measure, per task:
  * distances between subject centroids, and within-subject dispersion
  * their ratio, and a leave-one-subject-out centroid distance
  * silhouette score by subject (in 2-D PCA space and in the full embedding)
with bootstrap 95% CIs (subjects resampled for centroid metrics, trials
resampled within subject for silhouettes).

Outputs (in --outdir):
    read_pca_shared_axes.png, imagine_pca_shared_axes.png
    read_report.json, imagine_report.json
    read_centroid_distance_matrix.csv, imagine_centroid_distance_matrix.csv
    summary.json

    python -m analysis.subject_variance --outdir covcorr_report
"""

import argparse
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from eeg2text.data import StreamingEEGDataset
from eeg2text.eval_utils import write_json

TASKS = ("read", "imagine")
SUMMARY_METRICS = (
    "avg_between_subject_centroid_distance",
    "within_subject_dispersion_mean",
    "between_over_within_ratio",
    "leave_one_subject_out_distance_mean",
    "silhouette_score_2d_pca",
    "silhouette_score_full_embedding_sample",
)


# ---------- trial -> log-Euclidean correlation vector ----------

def correlation_matrix(X: np.ndarray, shrinkage: bool = True, eps: float = 1e-8) -> np.ndarray:
    """X: (C, T) -> (C, C) correlation matrix, regularised to stay positive definite."""
    X = X.astype(np.float64) - X.mean(axis=1, keepdims=True)
    cov = LedoitWolf().fit(X.T).covariance_ if shrinkage else (X @ X.T) / max(1, X.shape[1] - 1)
    d = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(d, d)
    corr = 0.5 * (corr + corr.T)
    return corr + eps * np.eye(len(corr))


def log_euclidean_vector(A: np.ndarray) -> np.ndarray:
    """Upper triangle of the matrix logarithm: a C(C+1)/2 vector."""
    L = logm(A).real
    L = 0.5 * (L + L.T)
    return L[np.triu_indices(len(L))].astype(np.float32)


def embed_batch(eeg, shrinkage: bool) -> np.ndarray:
    return np.stack([log_euclidean_vector(correlation_matrix(x, shrinkage)) for x in eeg.numpy()])


# ---------- two passes over the data ----------

def fit_pca_and_centroids(loaders, n_subjects, ipca, shrinkage):
    """Pass 1: fit one PCA on both tasks and compute each subject's mean embedding per task."""
    centroids = {}
    for task, loader in loaders.items():
        sums = [None] * n_subjects
        counts = np.zeros(n_subjects, dtype=int)
        for i, batch in enumerate(loader):
            emb = embed_batch(batch["eeg"], shrinkage)
            ipca.partial_fit(emb)
            for e, s in zip(emb.astype(np.float64), batch["subject_id"].numpy()):
                sums[s] = e.copy() if sums[s] is None else sums[s] + e
                counts[s] += 1
            if i % 50 == 0:
                print(f"[pass 1][{task}] batch {i}/{len(loader)}", flush=True)
        centroids[task] = [None if c == 0 else s / c for s, c in zip(sums, counts)]
    return centroids


def reservoir_add(reservoir: list, item, seen: int, capacity: int, rng) -> None:
    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        j = int(rng.integers(0, seen))
        if j < capacity:
            reservoir[j] = item


def sample_points(loader, ipca, shrinkage, seed, max_2d_per_subject, max_full_per_subject):
    """Pass 2: a random sample of trials per subject, in 2-D PCA space and in the full embedding."""
    rng = np.random.default_rng(seed)
    pts_2d, pts_full, seen = {}, {}, {}

    for batch in loader:
        emb = embed_batch(batch["eeg"], shrinkage)
        xy = ipca.transform(emb)[:, :2].astype(np.float32)
        for e, p, s in zip(emb, xy, batch["subject_id"].numpy()):
            s = int(s)
            seen[s] = seen.get(s, 0) + 1
            reservoir_add(pts_2d.setdefault(s, []), p, seen[s], max_2d_per_subject, rng)
            reservoir_add(pts_full.setdefault(s, []), e, seen[s], max_full_per_subject, rng)

    def stack(points):
        X = np.concatenate([np.stack(v) for v in points.values()])
        labels = np.concatenate([np.full(len(v), s) for s, v in points.items()])
        return X, labels

    return stack(pts_2d), stack(pts_full)


# ---------- statistics ----------

def nanmean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def safe_silhouette(X, labels) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def centroid_distance_matrix(centroids) -> np.ndarray:
    n = len(centroids)
    M = np.full((n, n), np.nan)
    present = [i for i, c in enumerate(centroids) if c is not None]
    for i in present:
        M[i, i] = 0.0
    for a, b in combinations(present, 2):
        M[a, b] = M[b, a] = np.linalg.norm(centroids[a] - centroids[b])
    return M


def subject_metrics(subjects, centroids, within):
    """Centroid-based metrics over a list of subject indices (repeats allowed, for the bootstrap).

    A subject paired with itself (possible in a bootstrap draw) contributes distance 0.
    """
    def dist(a, b):
        return 0.0 if a == b else float(np.linalg.norm(centroids[a] - centroids[b]))

    pairs = [dist(subjects[i], subjects[j]) for i in range(len(subjects)) for j in range(i + 1, len(subjects))]
    between = nanmean(pairs)
    within_mean = nanmean([within[s] for s in subjects])
    loso = nanmean([nanmean([dist(s, t) for j, t in enumerate(subjects) if j != i]) for i, s in enumerate(subjects)])
    return {
        "avg_between_subject_centroid_distance": between,
        "within_subject_dispersion_mean": within_mean,
        "between_over_within_ratio": between / within_mean if within_mean > 0 else float("nan"),
        "leave_one_subject_out_distance_mean": loso,
    }


def bootstrap_silhouette(X, labels, n_bootstrap, rng, max_points):
    """Resample trials within each subject; subsample large draws because silhouette is O(n^2)."""
    by_subject = [np.where(labels == s)[0] for s in np.unique(labels)]
    samples = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_subject])
        if len(idx) > max_points:
            idx = rng.choice(idx, size=max_points, replace=False)
        samples.append(safe_silhouette(X[idx], labels[idx]))
    return samples


def with_ci(estimate, samples, level):
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    alpha = (1 - level) / 2
    return {
        "estimate": float(estimate),
        "level": level,
        "low": float(np.quantile(samples, alpha)) if samples.size else float("nan"),
        "high": float(np.quantile(samples, 1 - alpha)) if samples.size else float("nan"),
        "n_bootstrap_valid": int(samples.size),
    }


def task_report(task, subject_codes, centroids, points_2d, points_full, args, seed):
    (xy, sid), (X, sid_full) = points_2d, points_full

    within = {}
    for s, c in enumerate(centroids):
        Xs = X[sid_full == s]
        within[s] = float(np.linalg.norm(Xs - c, axis=1).mean()) if c is not None and len(Xs) else float("nan")

    present = [s for s, c in enumerate(centroids) if c is not None and np.isfinite(within[s])]
    point = subject_metrics(present, centroids, within)
    point["silhouette_score_2d_pca"] = safe_silhouette(xy, sid)
    point["silhouette_score_full_embedding_sample"] = safe_silhouette(X, sid_full)

    rng = np.random.default_rng(seed)
    boot = {name: [] for name in point}
    for _ in range(args.n_bootstrap if len(present) >= 2 else 0):
        draw = list(rng.choice(present, size=len(present), replace=True))
        for name, value in subject_metrics(draw, centroids, within).items():
            boot[name].append(value)
    boot["silhouette_score_2d_pca"] = bootstrap_silhouette(
        xy, sid, args.n_bootstrap, np.random.default_rng(seed + 101), args.max_silhouette_ci_points)
    boot["silhouette_score_full_embedding_sample"] = bootstrap_silhouette(
        X, sid_full, args.n_bootstrap, np.random.default_rng(seed + 202), args.max_silhouette_ci_points)

    loso_by_subject = {s: nanmean([np.linalg.norm(c - o) for j, o in enumerate(centroids) if j != s and o is not None])
                       for s, c in enumerate(centroids) if c is not None}
    return {
        "task": task,
        "subjects": [{"subject_index": i, "subject_code": c} for i, c in enumerate(subject_codes)],
        **{name: with_ci(point[name], boot[name], args.ci_level) for name in point},
        "within_subject_dispersion_std": float(np.nanstd(list(within.values()))),
        "within_subject_dispersion_by_subject_index": within,
        "leave_one_subject_out_distance_by_subject_index": loso_by_subject,
    }


# ---------- plotting / saving ----------

def plot_pca(xy, labels, title, path, xlim, ylim):
    plt.figure()
    for s in np.unique(labels):
        m = labels == s
        plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.7, label=f"subject {s + 1}")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_matrix_csv(M, path, subject_codes):
    names = [f"sub-{code}" for code in subject_codes]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join([""] + names) + "\n")
        for name, row in zip(names, M):
            f.write(",".join([name] + ["" if not np.isfinite(v) else f"{v:.6f}" for v in row]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chisco-root", default="./ds005170/derivatives/preprocessed_pkl")
    ap.add_argument("--subject-codes", default="01,02,03,04,05")
    ap.add_argument("--outdir", default="covcorr_report")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pca-components", type=int, default=10)
    ap.add_argument("--no-shrinkage", action="store_true", help="Use the sample covariance instead of Ledoit-Wolf.")
    ap.add_argument("--max-plot-points-per-subject", type=int, default=2000)
    ap.add_argument("--max-fulld-per-subject", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=1000, help="0 disables confidence intervals.")
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--max-silhouette-ci-points", type=int, default=5000)
    args = ap.parse_args()

    shrinkage = not args.no_shrinkage
    subject_codes = [s.strip() for s in args.subject_codes.split(",") if s.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    loaders = {
        task: DataLoader(StreamingEEGDataset(args.chisco_root, task, subject_codes),
                         batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for task in TASKS
    }

    ipca = IncrementalPCA(n_components=args.pca_components)
    centroids = fit_pca_and_centroids(loaders, len(subject_codes), ipca, shrinkage)

    points = {
        task: sample_points(loader, ipca, shrinkage, args.seed + 1000 * (i + 1),
                            args.max_plot_points_per_subject, args.max_fulld_per_subject)
        for i, (task, loader) in enumerate(loaders.items())
    }

    all_xy = np.concatenate([points[t][0][0] for t in TASKS])
    pad = 0.05 * (all_xy.max(axis=0) - all_xy.min(axis=0) + 1e-6)
    xlim = (all_xy[:, 0].min() - pad[0], all_xy[:, 0].max() + pad[0])
    ylim = (all_xy[:, 1].min() - pad[1], all_xy[:, 1].max() + pad[1])

    summary = {
        "subjects": [{"subject_index": i, "subject_code": c} for i, c in enumerate(subject_codes)],
        "shared_pca_components": args.pca_components,
        "pca_explained_variance_ratio": ipca.explained_variance_ratio_.tolist(),
        "shrinkage": shrinkage,
        "confidence_interval_method": {
            "level": args.ci_level,
            "n_bootstrap": args.n_bootstrap,
            "subject_level_metrics": "bootstrap over subjects with replacement",
            "silhouette_metrics": "stratified bootstrap over sampled trials within each subject",
            "max_silhouette_ci_points": args.max_silhouette_ci_points,
        },
    }

    for i, task in enumerate(TASKS):
        points_2d, points_full = points[task]
        report = task_report(task, subject_codes, centroids[task], points_2d, points_full, args,
                             seed=args.seed + 3000 + 1000 * i)
        write_json(os.path.join(args.outdir, f"{task}_report.json"), report)
        save_matrix_csv(centroid_distance_matrix(centroids[task]),
                        os.path.join(args.outdir, f"{task}_centroid_distance_matrix.csv"), subject_codes)
        plot_pca(*points_2d, f"CHISCO {task}: log-correlation embeddings (shared PCA)",
                 os.path.join(args.outdir, f"{task}_pca_shared_axes.png"), xlim, ylim)
        summary[task] = {name: report[name] for name in SUMMARY_METRICS}

    write_json(os.path.join(args.outdir, "summary.json"), summary)
    print(f"Saved results to {args.outdir}\n")

    for name in SUMMARY_METRICS:
        for task in TASKS:
            m = summary[task][name]
            print(f"{task:8s} {name}: {m['estimate']:.4f} [{m['low']:.4f}, {m['high']:.4f}]")


if __name__ == "__main__":
    main()

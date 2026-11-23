"""Grad-CAM and input-gradient saliency on the test set.

For each test trial the score is the cosine similarity between the EEG
embedding and its target sentence embedding. We record:
  * temporal Grad-CAM on the BrainModule output, upsampled to the input length
  * channel saliency: mean |d score / d EEG| over time, per electrode
  * (optional) the same saliency in time slices, drawn as scalp topomaps

Single run -> plots in <run-dir>/plots/gradcam/
Many runs  -> plots in <runs-root>/plots/gradcam_aggregate/<task>/, pooled over runs of each task

Topomaps need MNE, resources/montage.csv and a text file listing the 122 model
input channels in order (--channels-txt).

    python -m analysis.gradcam --runs-root experiments/eeg2text --channels-txt channels.txt
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg2text.eval_utils import Run, find_run_dirs, get_device, read_json
from eeg2text.model import MODEL_DEFAULTS


def build_mne_info(channels_txt: str, montage_csv: str, sfreq: float):
    """MNE Info with electrode positions, needed for topomaps."""
    import mne

    with open(channels_txt, "r", encoding="utf-8") as f:
        ch_names = [line.strip() for line in f if line.strip()]
    with open(montage_csv, "r", encoding="utf-8") as f:
        positions = {row["label"]: (float(row["x"]), float(row["y"]), float(row["z"])) for row in csv.DictReader(f)}

    missing = [ch for ch in ch_names if ch not in positions]
    if missing:
        raise ValueError(f"{len(missing)} channels are missing from {montage_csv}, e.g. {missing[:10]}")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos={ch: positions[ch] for ch in ch_names}, coord_frame="head"))
    return info


def time_slices(n_samples: int, n_slices: int) -> List[Tuple[int, int]]:
    slice_len = math.ceil(n_samples / n_slices)
    return [(lo, min(n_samples, lo + slice_len)) for lo in range(0, n_samples, slice_len)]


class TemporalGradCAM:
    """Hooks the BrainModule output to get temporal Grad-CAM and input-gradient saliency."""

    def __init__(self, model):
        if getattr(model, "brain", None) is None:
            raise ValueError("Grad-CAM needs a model with a BrainModule (arch=brainmodule, use_brainmodule=True).")
        self.model = model
        self._z = None
        self._hook = model.brain.register_forward_hook(self._save_output)

    def _save_output(self, module, inputs, output):
        self._z = output                       # (B, latent_dim, T')
        self._z.retain_grad()

    def remove(self):
        self._hook.remove()

    def __call__(self, eeg, subject_ids, text):
        """Returns cam (B, T') scaled to [0, 1] per trial, and |d score / d EEG| (B, C, T)."""
        self.model.zero_grad(set_to_none=True)
        eeg = eeg.requires_grad_(True)

        score = (self.model(eeg, subject_ids) * F.normalize(text, dim=-1)).sum()
        score.backward()

        z, grad = self._z, self._z.grad
        cam = F.relu((z * grad.mean(dim=1, keepdim=True)).sum(dim=1))
        cam_min = cam.min(dim=-1, keepdim=True).values
        cam_max = cam.max(dim=-1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        self._z = None
        return cam.detach(), eeg.grad.abs().detach()


class SaliencyStats:
    """Running sums for the mean and SEM of each saliency map, over trials (and runs)."""

    def __init__(self):
        self.n = 0
        self.sums = defaultdict(float)
        self.sq_sums = defaultdict(float)

    def add(self, name: str, values: torch.Tensor) -> None:
        """values: (B, ...) for one batch of trials."""
        values = values.detach().cpu().double()
        self.sums[name] = self.sums[name] + values.sum(dim=0)
        self.sq_sums[name] = self.sq_sums[name] + (values ** 2).sum(dim=0)

    def merge(self, other: "SaliencyStats") -> None:
        self.n += other.n
        for name in other.sums:
            self.sums[name] = self.sums[name] + other.sums[name]
            self.sq_sums[name] = self.sq_sums[name] + other.sq_sums[name]

    def mean_sem(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.sums[name] / self.n
        var = torch.clamp(self.sq_sums[name] / self.n - mean ** 2, min=0.0)
        return mean.numpy(), (torch.sqrt(var) / math.sqrt(self.n)).numpy()


def process_run(run_dir: str, args, device) -> Tuple[str, SaliencyStats]:
    run = Run(run_dir, device, checkpoint_path=os.path.join(run_dir, f"{args.checkpoint}_checkpoint.pt"))
    sent2emb = read_json(run.sent_emb_path)
    model = run.load_model(text_dim=len(next(iter(sent2emb.values()))))
    dataset = run.test_dataset(sent2emb=sent2emb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    gradcam = TemporalGradCAM(model)
    stats = SaliencyStats()
    slices = None
    name = os.path.basename(run_dir)
    print(f"[{name}] task={run.task} | {len(dataset)} test trials")

    for i, batch in enumerate(loader):
        if args.max_batches and i >= args.max_batches:
            break
        eeg = batch["eeg"].to(device)
        cam, grad_abs = gradcam(eeg, batch["subject_id"].to(device), batch["target"].to(device))

        n_samples = eeg.shape[-1]
        cam = F.interpolate(cam.unsqueeze(1), size=n_samples, mode="linear", align_corners=False).squeeze(1)
        stats.n += len(eeg)
        stats.add("temporal", cam)
        stats.add("channel", grad_abs.mean(dim=-1))

        if slices is None:
            slices = time_slices(n_samples, args.n_time_slices)
        stats.add("spatiotemporal", torch.stack([grad_abs[:, :, lo:hi].mean(dim=-1) for lo, hi in slices], dim=1))

        if i % 10 == 0:
            print(f"[{name}] batch {i + 1}/{len(loader)}")

    gradcam.remove()
    return run.task, stats


def plot_topomap(values, info, ax, vmin, vmax):
    import mne
    mne.viz.plot_topomap(values, info, axes=ax, show=False, vlim=(vmin, vmax), cmap="RdBu_r", contours=0,
                         res=256, image_interp="cubic", extrapolate="head", sensors=True)


def save_plots(stats: SaliencyStats, out_dir: str, label: str, fs: float, n_time_slices: int, info) -> None:
    os.makedirs(out_dir, exist_ok=True)
    title_suffix = f"test set, N={stats.n} [{label}]"

    # Temporal Grad-CAM
    cam_mean, cam_sem = stats.mean_sem("temporal")
    t = np.arange(len(cam_mean)) / fs
    plt.figure()
    plt.plot(t, cam_mean, label="Mean temporal attribution")
    plt.fill_between(t, cam_mean - cam_sem, cam_mean + cam_sem, alpha=0.2, label="±1 SEM")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Importance (0–1)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradcam_temporal_mean.png"))
    plt.close()
    np.save(os.path.join(out_dir, "gradcam_temporal_mean.npy"), cam_mean)
    np.save(os.path.join(out_dir, "gradcam_temporal_time_seconds.npy"), t)

    # Channel saliency
    chan_mean, chan_sem = stats.mean_sem("channel")
    channels = np.arange(len(chan_mean))
    plt.figure()
    plt.bar(channels, chan_mean)
    plt.errorbar(channels, chan_mean, yerr=chan_sem, fmt="none", capsize=2)
    plt.title(f"Channel saliency — {title_suffix}")
    plt.xlabel("EEG channel index")
    plt.ylabel("mean |∂score/∂EEG| over time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradcam_channel_saliency.png"))
    plt.close()
    np.save(os.path.join(out_dir, "gradcam_channel_saliency.npy"), chan_mean)

    if info is None:
        print("Skipping topomaps (pass --channels-txt to enable).")
        return

    # Scalp topomap of channel saliency
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    plot_topomap(chan_mean, info, ax, float(chan_mean.min()), float(np.percentile(chan_mean, 99)))
    ax.set_title(f"Channel saliency — {title_suffix}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradcam_channel_topomap.png"), dpi=200)
    plt.close(fig)

    # Topomaps per time slice, on a shared colour scale
    topo_mean, _ = stats.mean_sem("spatiotemporal")          # (n_slices, C)
    slices = time_slices(len(cam_mean), n_time_slices)
    vmin, vmax = float(topo_mean.min()), float(np.percentile(topo_mean, 99))
    ncols = min(5, len(slices))
    nrows = math.ceil(len(slices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, values, (lo, hi) in zip(axes, topo_mean, slices):
        plot_topomap(values, info, ax, vmin, vmax)
        ax.set_title(f"{lo / fs:.2f}–{hi / fs:.2f}s", fontsize=10)
    for ax in axes[len(slices):]:
        ax.axis("off")
    fig.suptitle(f"Input-gradient topomaps over time — {title_suffix}", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gradcam_spatiotemporal_topomaps.png"), dpi=200)
    plt.close(fig)
    np.save(os.path.join(out_dir, "gradcam_spatiotemporal_topomaps.npy"), topo_mean)
    np.save(os.path.join(out_dir, "gradcam_spatiotemporal_times_seconds.npy"), np.array(slices) / fs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--run-dir", help="A single run folder")
    where.add_argument("--runs-root", help="Folder searched recursively for run folders; results pooled per task")
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="Only use the first N batches (0 = all).")
    ap.add_argument("--fs", type=float, default=500.0, help="Sampling rate (Hz).")
    ap.add_argument("--n-time-slices", type=int, default=15, help="Number of time slices for the topomap grid.")
    ap.add_argument("--montage-csv", default="resources/montage.csv")
    ap.add_argument("--channels-txt", default=None,
                    help="Channel names, one per line, in the same order as the model's 122 input channels.")
    args = ap.parse_args()

    device = get_device(args.device)
    info = build_mne_info(args.channels_txt, args.montage_csv, args.fs) if args.channels_txt else None

    if args.run_dir:
        task, stats = process_run(args.run_dir, args, device)
        save_plots(stats, os.path.join(args.run_dir, "plots", "gradcam"), task, args.fs, args.n_time_slices, info)
        return

    by_task = {}
    for run_dir in find_run_dirs(args.runs_root, f"{args.checkpoint}_checkpoint.pt"):
        config = {**MODEL_DEFAULTS, **read_json(os.path.join(run_dir, "config.json"))}
        if config["arch"] != "brainmodule" or not config["use_brainmodule"]:
            print(f"Skipping {run_dir}: no BrainModule")
            continue
        task, stats = process_run(run_dir, args, device)
        if task in by_task:
            by_task[task].merge(stats)
        else:
            by_task[task] = stats

    for task, stats in by_task.items():
        out_dir = os.path.join(args.runs_root, "plots", "gradcam_aggregate", task)
        save_plots(stats, out_dir, task, args.fs, args.n_time_slices, info)
        print(f"Saved {task} plots ({stats.n} trials) to {out_dir}")


if __name__ == "__main__":
    main()

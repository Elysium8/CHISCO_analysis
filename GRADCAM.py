import os
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import pandas as pd
    import mne
except Exception:
    pd = None
    mne = None


sys.path.append("./ds005170")

from updated_train_eeg_to_text_RAM import EEGToTextEmbedModel  # type: ignore
from data_loader import ChiscoStreamingDataset  # type: ignore


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sentence_embeddings(path: str) -> Dict[str, np.ndarray]:
    """Read sentence embeddings into float32 arrays."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        out[k] = np.asarray(v, dtype=np.float32)
    return out


def cfg_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    if default is not None:
        return default
    raise KeyError(f"None of these keys found in config.json: {keys}")


def normalize_sentence(s):
    return str(s).strip()


def indices_from_sentence_split(sentences, split_sentences):
    split_set = set(normalize_sentence(s) for s in split_sentences)
    return [
        i for i, s in enumerate(sentences)
        if normalize_sentence(s) in split_set
    ]


def upsample_cam_to_input_length(cam_Tprime: torch.Tensor, T: int) -> torch.Tensor:
    """Stretch the Grad-CAM sequence back to the original EEG length."""
    cam = cam_Tprime.unsqueeze(1)
    cam_up = F.interpolate(cam, size=T, mode="linear", align_corners=False)
    return cam_up.squeeze(1)


def read_channel_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f.readlines()]
    return [n for n in names if n]


def build_info_from_montage(
    channels_txt: str,
    montage_csv: str,
    sfreq: float,
) -> "mne.Info":
    """Build the MNE channel metadata needed for topomap plots."""
    if mne is None or pd is None:
        raise RuntimeError("Topomap requires mne + pandas. Install with: pip install mne pandas")

    ch_names = read_channel_list(channels_txt)

    df = pd.read_csv(montage_csv)
    if not {"label", "x", "y", "z"}.issubset(set(df.columns)):
        raise ValueError(f"montage.csv must have columns label,x,y,z. Got: {list(df.columns)}")

    pos_all = {
        str(r["label"]): (float(r["x"]), float(r["y"]), float(r["z"]))
        for _, r in df.iterrows()
    }

    missing = [ch for ch in ch_names if ch not in pos_all]
    if missing:
        raise ValueError(
            f"{len(missing)} channels from channels.txt are missing in montage.csv. "
            f"First few: {missing[:20]}"
        )

    ch_pos = {ch: pos_all[ch] for ch in ch_names}

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=["eeg"] * len(ch_names))
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage)
    return info


def compute_time_slices(
    T: int,
    fs: float,
    timeslice_sec: float,
    timeslice_count: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Split an EEG recording into sample ranges and matching time ranges."""
    if timeslice_sec > 0:
        slice_len = max(1, int(round(timeslice_sec * fs)))
        K = int(math.ceil(T / slice_len))
    else:
        # Let the caller request a rough number of slices instead.
        K = max(1, int(timeslice_count))
        slice_len = int(math.ceil(T / K))

    slices: List[Tuple[int, int]] = []
    times_sec = []
    for k in range(K):
        lo = k * slice_len
        hi = min(T, (k + 1) * slice_len)
        if lo >= T:
            break
        slices.append((lo, hi))
        times_sec.append((lo / fs, hi / fs))

    return slices, np.asarray(times_sec, dtype=np.float64)


class TemporalGradCAM:
    """Small wrapper for temporal Grad-CAM and input-channel saliency."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._z: Optional[torch.Tensor] = None
        self._hook = model.brain.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # The brain module returns z with shape (B, latent_dim, T').
        self._z = out
        self._z.retain_grad()

    def remove(self):
        self._hook.remove()

    def compute(self, eeg: torch.Tensor, subj: torch.Tensor, text: torch.Tensor):
        """Return temporal CAM, cosine scores, channel saliency, and raw gradients."""
        self.model.zero_grad(set_to_none=True)

        # We need gradients at the input for the channel and topomap views.
        eeg = eeg.requires_grad_(True)

        # Compare the EEG embedding with its target text embedding.
        eeg_emb = self.model(eeg, subj)
        text_n = F.normalize(text, dim=-1)

        score_per_sample = (eeg_emb * text_n).sum(dim=-1)
        score = score_per_sample.sum()
        score.backward()

        # Temporal Grad-CAM on the brain-module output.
        z = self._z
        if z is None or z.grad is None:
            raise RuntimeError("Grad-CAM: missing z or z.grad. Did the hook run?")

        g = z.grad
        w = g.mean(dim=1)
        cam_Tprime = (z * w.unsqueeze(1)).sum(dim=1)
        cam_Tprime = F.relu(cam_Tprime)

        # Keep each sample on the same 0-1 scale before averaging.
        cam_min = cam_Tprime.min(dim=-1, keepdim=True).values
        cam_max = cam_Tprime.max(dim=-1, keepdim=True).values
        cam_Tprime = (cam_Tprime - cam_min) / (cam_max - cam_min + 1e-8)

        # Spatial saliency from gradients on the input EEG.
        if eeg.grad is None:
            raise RuntimeError("Expected eeg.grad but got None.")
        grad_abs_time = eeg.grad.abs()
        chan_imp = grad_abs_time.mean(dim=-1)

        # Drop the stored activation so the next batch starts clean.
        self._z = None

        return cam_Tprime.detach(), score_per_sample.detach(), chan_imp.detach(), grad_abs_time.detach()


def _is_run_dir(path: str, checkpoint_name: str) -> bool:
    if not os.path.isdir(path):
        return False
    config_path = os.path.join(path, "config.json")
    split_path = os.path.join(path, "split.json")
    ckpt_path = os.path.join(path, checkpoint_name)
    return os.path.exists(config_path) and os.path.exists(split_path) and os.path.exists(ckpt_path)


def _discover_run_dirs(runs_root: str, checkpoint_name: str) -> List[str]:
    runs: List[str] = []
    if not os.path.isdir(runs_root):
        return runs

    # Most run roots are laid out as direct children.
    for name in sorted(os.listdir(runs_root)):
        p = os.path.join(runs_root, name)
        if _is_run_dir(p, checkpoint_name):
            runs.append(p)

    # Some sweeps add one extra folder level.
    if len(runs) == 0:
        for name in sorted(os.listdir(runs_root)):
            p = os.path.join(runs_root, name)
            if not os.path.isdir(p):
                continue
            for name2 in sorted(os.listdir(p)):
                p2 = os.path.join(p, name2)
                if _is_run_dir(p2, checkpoint_name):
                    runs.append(p2)

    return runs


def _normalize_task_name(task: Any) -> str:
    t = str(task).strip().lower()
    if "imag" in t:
        return "imagine"
    if "read" in t:
        return "read"
    return t


def _accum_add(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["cam_sum"] += src["cam_sum"]
    dst["cam_sq_sum"] += src["cam_sq_sum"]
    dst["chan_sum"] += src["chan_sum"]
    dst["chan_sq_sum"] += src["chan_sq_sum"]
    dst["n_total"] += int(src["n_total"])

    if dst.get("topo_sum") is not None and src.get("topo_sum") is not None:
        dst["topo_sum"] += src["topo_sum"]
        dst["topo_sq_sum"] += src["topo_sq_sum"]


def _process_single_run(
    run_dir: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Run Grad-CAM for one experiment folder and return CPU accumulators."""
    config_path = os.path.join(run_dir, "config.json")
    split_path = os.path.join(run_dir, "split.json")
    ckpt_name = "best_checkpoint.pt" if args.checkpoint == "best" else "last_checkpoint.pt"
    ckpt_path = os.path.join(run_dir, ckpt_name)

    cfg = load_json(config_path)
    split_info = load_json(split_path)

    # Pull the training settings back out of the saved config.
    chisco_root = cfg_get(cfg, "chisco_root")
    sent_emb_path = cfg_get(cfg, "sent_emb_path")
    task = cfg_get(cfg, "task")
    subject_codes = cfg_get(cfg, "subject_codes")

    use_eegnet = cfg_get(cfg, "use_eegnet")
    eegnet_temporal_kern = cfg_get(cfg, "eegnet_temporal_kern")
    eegnet_F1 = cfg_get(cfg, "eegnet_F1", "eegnet_f1")
    eegnet_D = cfg_get(cfg, "eegnet_D", "eegnet_d")
    raw_frontend_out_ch = cfg_get(cfg, "raw_frontend_out_ch", default=32)

    use_spatial_attn = cfg_get(cfg, "use_spatial_attn")
    use_subject_layer = cfg_get(cfg, "use_subject_layer")
    use_residual_blocks = cfg_get(cfg, "use_residual_blocks")

    hidden_channels = cfg_get(cfg, "hidden_channels")
    final_hidden = cfg_get(cfg, "final_hidden")
    n_blocks = cfg_get(cfg, "n_blocks")
    latent_dim = cfg_get(cfg, "latent_dim")
    pooling = cfg_get(cfg, "pooling")

    # Sentence embeddings are stored as JSON lists.
    sent2emb = load_sentence_embeddings(sent_emb_path)
    if len(sent2emb) == 0:
        raise RuntimeError(f"No sentence embeddings found at {sent_emb_path}")
    text_dim = int(next(iter(sent2emb.values())).shape[0])

    # Use the same dataset setup as training, but keep augmentation off.
    base_ds = ChiscoStreamingDataset(
        root=chisco_root,
        task=task,
        subject_codes=subject_codes,
        sent2emb=sent2emb,
        cache_runs=2,
        augment=False,
        noise_std=float(cfg.get("noise_std", 0.01)),
        jitter_max=int(cfg.get("jitter_max", 10)),
        channel_dropout_p=float(cfg.get("channel_dropout_p", 0.05)),
    )

    # The split file may store either sentences or raw indices.
    if "test_sentences" in split_info:
        val_idx = indices_from_sentence_split(
            base_ds.sentences,
            split_info["test_sentences"],
        )
    elif "test_idx" in split_info:
        val_idx = split_info["test_idx"]
    else:
        raise KeyError("split.json must contain either test_sentences or test_idx")

    if not val_idx:
        raise ValueError("Test split produced no samples.")

    # Fail early if the saved split no longer matches the dataset order.
    n_samples = len(base_ds)
    if any((i < 0 or i >= n_samples) for i in val_idx):
        raise ValueError("val_idx contains out-of-range indices for the current dataset ordering.")

    sample0 = base_ds[0]
    n_channels = int(sample0["eeg"].shape[0])
    n_subjects = len(subject_codes)

    from torch.utils.data import DataLoader, Subset
    val_dataset = Subset(base_ds, val_idx)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    # Recreate the exact model shape before loading the checkpoint.
    model = EEGToTextEmbedModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        text_dim=text_dim,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        final_hidden=final_hidden,
        n_blocks=n_blocks,
        pooling=pooling,
        use_eegnet=use_eegnet,
        eegnet_temporal_kern=int(eegnet_temporal_kern),
        eegnet_F1=int(eegnet_F1),
        eegnet_D=int(eegnet_D),
        use_spatial_attn=bool(use_spatial_attn),
        use_subject_layer=bool(use_subject_layer),
        use_residual_blocks=bool(use_residual_blocks),
        raw_frontend_out_ch=int(raw_frontend_out_ch),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Accumulate enough statistics to save means and SEM later.
    cammer = TemporalGradCAM(model)

    cam_sum = None
    cam_sq_sum = None
    chan_sum = None
    chan_sq_sum = None
    n_total = 0
    T_in_ref = None

    # Optional topomap statistics, one row per time slice.
    slices: Optional[List[Tuple[int, int]]] = None
    slice_times_sec: Optional[np.ndarray] = None
    topo_sum = None  # (K, C), float64 on CPU
    topo_sq_sum = None

    max_batches = int(args.max_batches)

    print(f"[{os.path.basename(run_dir)}] task={task} | Val batches: {len(val_loader)} (batch_size={args.batch_size})")
    with torch.enable_grad():
        for bi, batch in enumerate(val_loader):
            if max_batches > 0 and bi >= max_batches:
                break

            if bi % 10 == 0:
                print(f"[{os.path.basename(run_dir)}] Grad-CAM batch {bi}/{len(val_loader)} | samples so far: {n_total}")

            eeg = batch["eeg"].to(device)
            subj = batch["subject_id"].to(device)
            text = batch["target"].to(device)

            cam_Tprime, sim, chan_imp, grad_abs_time = cammer.compute(eeg, subj, text)

            T_in = int(eeg.shape[-1])
            if T_in_ref is None:
                T_in_ref = T_in

                # Initialize time slices after seeing the true input length.
                slices, slice_times_sec = compute_time_slices(
                    T=T_in_ref,
                    fs=float(args.fs),
                    timeslice_sec=float(args.timeslice_sec),
                    timeslice_count=int(args.timeslice_count),
                )
                K = len(slices)
                topo_sum = torch.zeros((K, n_channels), dtype=torch.float64)
                topo_sq_sum = torch.zeros((K, n_channels), dtype=torch.float64)

                if args.timeslice_sec > 0:
                    print(f"[{os.path.basename(run_dir)}] Spatiotemporal: slice {args.timeslice_sec:.3f}s -> {K} slices")
                else:
                    print(f"[{os.path.basename(run_dir)}] Spatiotemporal: target {args.timeslice_count} -> {K} slices")

            elif T_in != T_in_ref:
                raise RuntimeError(f"Varying T in val set: {T_in} vs {T_in_ref}")

            cam_up = upsample_cam_to_input_length(cam_Tprime, T_in)

            B = int(cam_up.shape[0])
            n_total += B

            # Temporal average and variance terms.
            cam_batch_sum = cam_up.sum(dim=0).detach().cpu()
            cam_batch_sq_sum = (cam_up ** 2).sum(dim=0).detach().cpu()

            if cam_sum is None:
                cam_sum = cam_batch_sum.clone()
                cam_sq_sum = cam_batch_sq_sum.clone()
            else:
                cam_sum += cam_batch_sum
                cam_sq_sum += cam_batch_sq_sum

            # Channel-level saliency, averaged over time inside compute().
            chan_batch_sum = chan_imp.sum(dim=0).detach().cpu()
            chan_batch_sq_sum = (chan_imp ** 2).sum(dim=0).detach().cpu()

            if chan_sum is None:
                chan_sum = chan_batch_sum.clone()
                chan_sq_sum = chan_batch_sq_sum.clone()
            else:
                chan_sum += chan_batch_sum
                chan_sq_sum += chan_batch_sq_sum

            # Time-resolved input-gradient topomaps.
            if (not args.no_spatiotemporal) and (topo_sum is not None) and (slices is not None):
                for k, (lo, hi) in enumerate(slices):
                    slice_mean = grad_abs_time[:, :, lo:hi].mean(dim=-1)
                    topo_sum[k] += slice_mean.sum(dim=0).detach().cpu().to(torch.float64)
                    topo_sq_sum[k] += (slice_mean ** 2).sum(dim=0).detach().cpu().to(torch.float64)

            # Free a few large tensors promptly, especially useful on Windows.
            del eeg, subj, text, cam_Tprime, cam_up, chan_imp, sim, grad_abs_time

            if device.type == "cuda" and bi % 20 == 0:
                torch.cuda.empty_cache()

    cammer.remove()

    if cam_sum is None or cam_sq_sum is None or chan_sum is None or chan_sq_sum is None:
        raise RuntimeError(f"No batches processed for run: {run_dir}")

    return {
        "task_name": _normalize_task_name(task),
        "run_dir": run_dir,
        "n_total": int(n_total),
        "T_in_ref": int(T_in_ref) if T_in_ref is not None else None,
        "n_channels": int(n_channels),

        "cam_sum": cam_sum.to(torch.float64),
        "cam_sq_sum": cam_sq_sum.to(torch.float64),
        "chan_sum": chan_sum.to(torch.float64),
        "chan_sq_sum": chan_sq_sum.to(torch.float64),

        "slices": slices,
        "slice_times_sec": slice_times_sec,
        "topo_sum": topo_sum,
        "topo_sq_sum": topo_sq_sum
    }


def _save_aggregate_plots(
    out_dir: str,
    agg: Dict[str, Any],
    args: argparse.Namespace,
    task_label: str,
    info_montage: Optional["mne.Info"],
    can_topomap: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    n_total = int(agg["n_total"])
    cam_sum = agg["cam_sum"]
    cam_sq_sum = agg["cam_sq_sum"]
    chan_sum = agg["chan_sum"]
    chan_sq_sum = agg["chan_sq_sum"]

    # Temporal mean and SEM.
    cam_mean = cam_sum / max(n_total, 1)
    cam_var = (cam_sq_sum / max(n_total, 1)) - (cam_mean ** 2)
    cam_var = torch.clamp(cam_var, min=0.0)
    cam_std = torch.sqrt(cam_var)
    cam_sem = cam_std / (max(n_total, 1) ** 0.5)

    # Spatial mean and SEM.
    chan_mean = chan_sum / max(n_total, 1)
    chan_var = (chan_sq_sum / max(n_total, 1)) - (chan_mean ** 2)
    chan_var = torch.clamp(chan_var, min=0.0)
    chan_std = torch.sqrt(chan_var)
    chan_sem = chan_std / (max(n_total, 1) ** 0.5)

    # Save the temporal curve and its time axis.
    T = int(cam_mean.shape[0])
    x_sec = np.arange(T) / float(args.fs)

    temporal_png = os.path.join(out_dir, "gradcam_temporal_val_mean.png")
    temporal_npy = os.path.join(out_dir, "gradcam_temporal_val_mean.npy")
    temporal_t = os.path.join(out_dir, "gradcam_temporal_time_seconds.npy")

    plt.figure()
    plt.plot(x_sec, cam_mean.numpy(), label=f"Mean temporal attribution")
    plt.fill_between(
        x_sec,
        (cam_mean - cam_sem).numpy(),
        (cam_mean + cam_sem).numpy(),
        alpha=0.2,
        label="±1 SEM",
    )
    plt.xlabel(f"Time (s) @ {args.fs:g} Hz", fontsize=14)
    plt.ylabel("Importance (0–1)", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(temporal_png)
    plt.close()

    np.save(temporal_npy, cam_mean.numpy())
    np.save(temporal_t, x_sec)

    # Save the channel bar plot and raw values.
    spatial_png = os.path.join(out_dir, "gradcam_spatial_channel_val_mean.png")
    spatial_npy = os.path.join(out_dir, "gradcam_spatial_channel_val_mean.npy")

    ch = np.arange(int(chan_mean.shape[0]))

    plt.figure()
    plt.bar(ch, chan_mean.numpy())
    plt.errorbar(ch, chan_mean.numpy(), yerr=chan_sem.numpy(), fmt="none", capsize=2)
    plt.title(f"Spatial saliency — Validation Average (N={n_total}) [{task_label}]")
    plt.xlabel("EEG channel index")
    plt.ylabel("mean(|∂score/∂EEG|) over time")
    plt.tight_layout()
    plt.savefig(spatial_png)
    plt.close()

    np.save(spatial_npy, chan_mean.numpy())

    if can_topomap and (info_montage is not None):
        topomap_png = os.path.join(out_dir, "gradcam_spatial_topomap.png")
        try:
            v = chan_mean.numpy()

            vmin = float(np.min(v))
            vmax = float(np.percentile(v, 99))
            if vmax <= vmin:
                vmax = vmin + 1e-8

            fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.8))
            try:
                mne.viz.plot_topomap(
                    v,
                    info_montage,
                    axes=ax,
                    show=False,
                    vlim=(vmin, vmax),
                    cmap="RdBu_r",
                    contours=0,
                    res=256,
                    image_interp="cubic",
                    extrapolate="head",
                    sensors=True,
                )
            except TypeError:
                mne.viz.plot_topomap(
                    v,
                    info_montage,
                    axes=ax,
                    show=False,
                    contours=0,
                )

            ax.set_title(f"Spatial saliency — Val Avg (N={n_total}) [{task_label}]", fontsize=10)

            plt.tight_layout()
            plt.savefig(topomap_png, dpi=200)
            plt.close(fig)
            print("Saved spatial topomap:", topomap_png)

        except Exception as e:
            print("WARNING: Failed to create spatial topomap:", repr(e))
    else:
        print("Skipping spatial topomap (provide --montage-csv and --channels-txt to enable).")

    topo_sum = agg.get("topo_sum", None)
    topo_sq_sum = agg.get("topo_sq_sum", None)
    slice_times_sec = agg.get("slice_times_sec", None)

    if (not args.no_spatiotemporal) and can_topomap and (info_montage is not None) and (topo_sum is not None):
        K = topo_sum.shape[0]
        topo_mean = (topo_sum / max(n_total, 1)).numpy()
        topo_var = (topo_sq_sum / max(n_total, 1)) - torch.from_numpy(topo_mean).to(torch.float64) ** 2
        topo_var = torch.clamp(topo_var, min=0.0)
        topo_sem = (torch.sqrt(topo_var) / (max(n_total, 1) ** 0.5)).numpy()  # noqa: F841

        topo_npy = os.path.join(out_dir, "gradcam_spatiotemporal_topomaps.npy")
        topo_times_npy = os.path.join(out_dir, "gradcam_spatiotemporal_times_seconds.npy")
        np.save(topo_npy, topo_mean)
        np.save(topo_times_npy, slice_times_sec)

        # Plot all time slices with a shared scale.
        vmin = float(np.min(topo_mean))
        vmax = float(np.percentile(topo_mean, 99))
        if vmax <= vmin:
            vmax = vmin + 1e-8

        ncols = int(min(5, K))
        nrows = int(math.ceil(K / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows))
        if isinstance(axes, np.ndarray):
            axes = axes.ravel()
        else:
            axes = [axes]

        for k in range(K):
            ax = axes[k]
            t0, t1 = slice_times_sec[k]
            title = f"{t0:.2f}–{t1:.2f}s"
            try:
                mne.viz.plot_topomap(
                    topo_mean[k],
                    info_montage,
                    axes=ax,
                    show=False,
                    vlim=(vmin, vmax),
                    cmap="RdBu_r",
                    contours=0,
                    res=256,
                    image_interp="cubic",
                    extrapolate="head",
                    sensors=True,
                )
            except TypeError:
                mne.viz.plot_topomap(
                    topo_mean[k],
                    info_montage,
                    axes=ax,
                    show=False,
                    contours=0,
                )
            ax.set_title(title, fontsize=10)

        for k in range(K, len(axes)):
            axes[k].axis("off")

        grid_png = os.path.join(out_dir, "gradcam_spatiotemporal_topomaps.png")
        fig.suptitle(f"Spatiotemporal input-gradient topomaps — Val Avg (N={n_total}) [{task_label}]", y=0.98)
        plt.tight_layout()
        plt.savefig(grid_png, dpi=200)
        plt.close(fig)

        print("Saved spatiotemporal topomap grid:", grid_png)
        print("Saved spatiotemporal numpy:", topo_npy)
        print("Saved spatiotemporal times:", topo_times_npy)
    else:
        if args.no_spatiotemporal:
            print("Skipping spatiotemporal topomaps (--no-spatiotemporal).")
        else:
            print("Skipping spatiotemporal topomaps (need --montage-csv and --channels-txt, and working montage).")

    print("\nDone.")
    print("Saved temporal plot:", temporal_png)
    print("Saved temporal mean:", temporal_npy)
    print("Saved temporal time axis:", temporal_t)
    print("Saved spatial plot:", spatial_png)
    print("Saved spatial mean:", spatial_npy)
    print("Output dir:", out_dir)


def main():
    p = argparse.ArgumentParser()

    # Keep both the old single-run path and the newer multi-run aggregation mode.
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=str, default=None,
                   help="Single experiment run folder (contains config.json, split.json, best/last checkpoint)")
    g.add_argument("--runs-root", type=str, default=None,
                   help="Directory containing subfolders of experiment runs. Will aggregate plots by task (read vs imagine).")

    p.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=8,
                   help="Grad-CAM uses backward; keep this small (e.g., 4-16).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Windows-friendly: use 0 to avoid DataLoader spawn overhead.")
    p.add_argument("--fs", type=float, default=500.0, help="Sampling rate Hz (you said 500).")
    p.add_argument("--pin-memory", action="store_true", default=False,
                   help="Usually leave OFF for analysis. Enable if you know it helps.")
    p.add_argument("--max-batches", type=int, default=0,
                   help="0 = entire val set. Otherwise, limit to first N batches for quicker runs.")

    # Topomap inputs.
    p.add_argument("--montage-csv", type=str, default="montage.csv",
                   help="Path to montage.csv with columns: label,x,y,z (enables topomaps).")
    p.add_argument("--channels-txt", type=str, default="channels.txt",
                   help="Path to channels.txt (one channel name per line, SAME order as model input channels).")

    # Spatiotemporal topomap controls.
    p.add_argument("--timeslice-sec", type=float, default=0,
                   help="Length of each time slice in seconds. If 0, uses --timeslice-count.")
    p.add_argument("--timeslice-count", type=int, default=15,
                   help="Target number of slices if --timeslice-sec is 0. Default 10.")
    p.add_argument("--no-spatiotemporal", action="store_true",
                   help="Disable spatiotemporal topomap grid even if montage+channels are provided.")

    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # Montage info is shared by all plots in this invocation.
    info_montage = None
    can_topomap = (args.montage_csv is not None) and (args.channels_txt is not None)
    if can_topomap:
        try:
            info_montage = build_info_from_montage(
                channels_txt=args.channels_txt,
                montage_csv=args.montage_csv,
                sfreq=float(args.fs),
            )
        except Exception as e:
            print("WARNING: montage/channels provided but failed to build Info:", repr(e))
            info_montage = None
            can_topomap = False

    ckpt_name = "best_checkpoint.pt" if args.checkpoint == "best" else "last_checkpoint.pt"

    # Mode 1: one run folder.
    if args.run_dir is not None:
        run_dir = args.run_dir
        if not _is_run_dir(run_dir, ckpt_name):
            raise FileNotFoundError(
                f"--run-dir does not look like a run folder (missing config/split/ckpt={ckpt_name}): {run_dir}"
            )

        print("Loading single run:", run_dir)
        single = _process_single_run(run_dir, args, device)

        out_dir = os.path.join(run_dir, "plots", "gradcam")
        _save_aggregate_plots(
            out_dir=out_dir,
            agg=single,
            args=args,
            task_label=single["task_name"],
            info_montage=info_montage,
            can_topomap=can_topomap,
        )
        return

    # Mode 2: many run folders, aggregated by task.
    runs_root = args.runs_root
    assert runs_root is not None

    run_dirs = _discover_run_dirs(runs_root, ckpt_name)
    if len(run_dirs) == 0:
        raise FileNotFoundError(
            f"No run subfolders found under --runs-root that contain "
            f"config.json, split.json, and {ckpt_name}: {runs_root}"
        )

    print(f"Discovered {len(run_dirs)} run folders under: {runs_root}")

    by_task: Dict[str, List[Dict[str, Any]]] = {"read": [], "imagine": []}
    skipped: List[Tuple[str, str]] = []

    # Process runs one by one so a bad run does not kill the whole aggregation.
    for rd in run_dirs:
        try:
            res = _process_single_run(rd, args, device)
            tname = res["task_name"]
            if tname not in by_task:
                # Only read/imagine runs are plotted here.
                skipped.append((rd, f"unknown task '{tname}'"))
                print(f"WARNING: skipping run (unknown task): {rd} | task={tname}")
                continue
            by_task[tname].append(res)
        except Exception as e:
            skipped.append((rd, repr(e)))
            print(f"WARNING: failed run, skipping: {rd} | err={repr(e)}")

    # Save one aggregate per task.
    for task_name in ["read", "imagine"]:
        runs = by_task[task_name]
        if len(runs) == 0:
            print(f"\nNo runs to aggregate for task='{task_name}'.")
            continue

        # Runs must be shape-compatible before their sums can be combined.
        ref = runs[0]
        T_ref = ref["T_in_ref"]
        C_ref = ref["n_channels"]

        slices_ref = ref.get("slices", None)
        slice_times_ref = ref.get("slice_times_sec", None)
        K_ref = None if slices_ref is None else len(slices_ref)

        # Start the aggregate from the first compatible run.
        agg: Dict[str, Any] = {
            "n_total": int(ref["n_total"]),
            "T_in_ref": int(T_ref) if T_ref is not None else None,
            "n_channels": int(C_ref),

            "cam_sum": ref["cam_sum"].clone(),
            "cam_sq_sum": ref["cam_sq_sum"].clone(),
            "chan_sum": ref["chan_sum"].clone(),
            "chan_sq_sum": ref["chan_sq_sum"].clone(),

            "slices": slices_ref,
            "slice_times_sec": slice_times_ref,
            "topo_sum": ref.get("topo_sum", None).clone() if ref.get("topo_sum", None) is not None else None,
            "topo_sq_sum": ref.get("topo_sq_sum", None).clone() if ref.get("topo_sq_sum", None) is not None else None,
        }

        used = 1
        for r in runs[1:]:
            if r["T_in_ref"] != T_ref:
                skipped.append((r["run_dir"], f"incompatible T: {r['T_in_ref']} vs {T_ref} (task={task_name})"))
                print(f"WARNING: skipping incompatible run (T mismatch) task={task_name}: {r['run_dir']}")
                continue
            if r["n_channels"] != C_ref:
                skipped.append((r["run_dir"], f"incompatible n_channels: {r['n_channels']} vs {C_ref} (task={task_name})"))
                print(f"WARNING: skipping incompatible run (C mismatch) task={task_name}: {r['run_dir']}")
                continue

            # Spatiotemporal outputs must have matching slices.
            if (agg.get("topo_sum") is not None) or (r.get("topo_sum") is not None):
                if (agg.get("topo_sum") is None) or (r.get("topo_sum") is None):
                    skipped.append((r["run_dir"], f"spatiotemporal availability mismatch (task={task_name})"))
                    print(f"WARNING: skipping incompatible run (spatiotemporal mismatch) task={task_name}: {r['run_dir']}")
                    continue
                if K_ref is not None and r.get("slices") is not None and len(r["slices"]) != K_ref:
                    skipped.append((r["run_dir"], f"incompatible slice count: {len(r['slices'])} vs {K_ref} (task={task_name})"))
                    print(f"WARNING: skipping incompatible run (slice count mismatch) task={task_name}: {r['run_dir']}")
                    continue

            _accum_add(agg, r)
            used += 1

        # Store aggregate outputs under the root rather than any single run.
        out_dir = os.path.join(runs_root, "plots", "gradcam_aggregate", task_name)
        print(f"\nAggregating task='{task_name}' from {used}/{len(runs)} runs -> {out_dir}")
        _save_aggregate_plots(
            out_dir=out_dir,
            agg=agg,
            args=args,
            task_label=task_name,
            info_montage=info_montage,
            can_topomap=can_topomap,
        )

    if skipped:
        print("\nSkipped runs:")
        for rd, reason in skipped:
            print(" -", rd, "|", reason)


if __name__ == "__main__":
    main()
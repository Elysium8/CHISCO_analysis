"""Full EEG -> sentence-embedding models and a single place to build them from a config."""

from typing import Any, Dict

import torch
import torch.nn as nn

from eeg2text.brain_module import BrainModule
from eeg2text.chisco_baseline import CHISCOBaselineEEGEncoder
from eeg2text.model_parts import EEGFrontEnd

# Default model hyperparameters. train.py uses these as its CLI defaults, and
# build_model() falls back to them for keys missing from a saved config.
MODEL_DEFAULTS: Dict[str, Any] = {
    "arch": "brainmodule",            # "brainmodule" or "chisco_baseline"
    # brainmodule architecture
    "latent_dim": 256,
    "hidden_channels": 256,
    "final_hidden": 256,
    "n_blocks": 5,
    "pooling": "mean",
    "use_frontend": True,
    "use_eegnet": True,
    "eegnet_temporal_kern": 64,
    "eegnet_f1": 16,
    "eegnet_d": 2,
    "raw_frontend_out_ch": 32,
    "use_brainmodule": True,
    "use_spatial_attn": True,
    "use_subject_layer": True,
    "use_residual_blocks": True,
    # CHISCO baseline architecture
    "baseline_layer": 0,
    "baseline_pooling": "mean",
    "baseline_size1": 8,
    "baseline_size2": 8,
    "baseline_feel1": 125,
    "baseline_feel2": 25,
    "baseline_dropout1": 0.1,
    "baseline_dropout2": 0.1,
    "baseline_latent_dim": 256,
}


class EEGToTextEmbedModel(nn.Module):
    """Front end (EEGNet-style or 1x1 conv) -> optional BrainModule -> pooling -> MLP head.

    Every stage can be switched off for ablations. Output is L2-normalised.
    """

    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        text_dim: int,
        latent_dim: int,
        hidden_channels: int,
        final_hidden: int,
        n_blocks: int,
        pooling: str,
        use_frontend: bool,
        use_eegnet: bool,
        eegnet_temporal_kern: int,
        eegnet_f1: int,
        eegnet_d: int,
        raw_frontend_out_ch: int,
        use_brainmodule: bool,
        use_spatial_attn: bool,
        use_subject_layer: bool,
        use_residual_blocks: bool,
    ):
        super().__init__()
        if pooling not in ("mean", "max"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling

        if not use_frontend:
            self.eeg_frontend = nn.Identity()
            frontend_out_ch = n_channels
        elif use_eegnet:
            self.eeg_frontend = EEGFrontEnd(n_channels, temporal_kern=eegnet_temporal_kern, F1=eegnet_f1, D=eegnet_d)
            frontend_out_ch = eegnet_f1 * eegnet_d
        else:
            self.eeg_frontend = nn.Conv1d(n_channels, raw_frontend_out_ch, kernel_size=1)
            frontend_out_ch = raw_frontend_out_ch

        if use_brainmodule:
            self.brain = BrainModule(
                in_channels=frontend_out_ch,
                out_channels=latent_dim,
                n_subjects=n_subjects,
                hidden_channels=hidden_channels,
                final_hidden=final_hidden,
                n_blocks=n_blocks,
                spatial_out_channels=frontend_out_ch,
                use_spatial_attn=use_spatial_attn,
                use_subject_layer=use_subject_layer,
                use_residual_blocks=use_residual_blocks,
            )
            # Learned 2-D "positions" for the front-end feature maps, used by spatial attention.
            self.sensor_pos = nn.Parameter(torch.randn(frontend_out_ch, 2) * 0.1)
            head_dim = latent_dim
        else:
            self.brain = None
            self.sensor_pos = None
            head_dim = frontend_out_ch

        self.proj = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, text_dim),
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        z = self.eeg_frontend(x)                    # (B, C', T')
        if self.brain is not None:
            z = self.brain(z, self.sensor_pos, subject_ids)
        z = z.mean(dim=-1) if self.pooling == "mean" else z.max(dim=-1).values
        return nn.functional.normalize(self.proj(z), dim=-1)


class CHISCOBaselineWrapper(nn.Module):
    """Gives the baseline the same model(eeg, subject_ids) signature; subject IDs are ignored."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_model(config: Dict[str, Any], n_channels: int, n_subjects: int, text_dim: int) -> nn.Module:
    """Build the model described by a run's config.json (or train.py's arguments)."""
    cfg = {**MODEL_DEFAULTS, **config}

    if cfg["arch"] == "chisco_baseline":
        return CHISCOBaselineWrapper(
            CHISCOBaselineEEGEncoder(
                n_channels=n_channels,
                text_dim=text_dim,
                latent_dim=cfg["baseline_latent_dim"],
                dropout1=cfg["baseline_dropout1"],
                dropout2=cfg["baseline_dropout2"],
                feel1=cfg["baseline_feel1"],
                feel2=cfg["baseline_feel2"],
                size1=cfg["baseline_size1"],
                size2=cfg["baseline_size2"],
                layer=cfg["baseline_layer"],
                pooling=cfg["baseline_pooling"],
            )
        )

    if cfg["arch"] != "brainmodule":
        raise ValueError(f"Unknown arch: {cfg['arch']}")

    return EEGToTextEmbedModel(
        n_channels=n_channels,
        n_subjects=n_subjects,
        text_dim=text_dim,
        **{k: cfg[k] for k in (
            "latent_dim", "hidden_channels", "final_hidden", "n_blocks", "pooling",
            "use_frontend", "use_eegnet", "eegnet_temporal_kern", "eegnet_f1", "eegnet_d",
            "raw_frontend_out_ch", "use_brainmodule", "use_spatial_attn", "use_subject_layer",
            "use_residual_blocks",
        )},
    )

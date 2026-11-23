"""Re-implementation of the CHISCO reference EEG encoder, used as a baseline.

EEGcnn -> (optional) Transformer encoder -> temporal pooling -> MLP -> L2 normalise
"""

import math

import torch
import torch.nn as nn


class EEGcnn(nn.Module):
    """(B, Chans, Samples) -> (B, F2, Samples // (P1 * P2))"""

    def __init__(
        self,
        Chans: int = 122,
        dropoutRate: float = 0.1,
        kernLength1: int = 125,
        kernLength2: int = 25,
        F1: int = 8,
        D: int = 8,
        F2: int = 64,
        P1: int = 2,
        P2: int = 5,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(dropoutRate),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, kernLength2), groups=D * F1, padding="same", bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(dropoutRate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(x.unsqueeze(1)))
        return x.squeeze(2)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for (B, T, D) inputs."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)])


class CHISCOBaselineEEGEncoder(nn.Module):
    """Maps EEG (B, C, T) to an L2-normalised embedding (B, text_dim)."""

    def __init__(
        self,
        n_channels: int,
        text_dim: int,
        latent_dim: int = 256,
        dropout1: float = 0.1,
        feel1: int = 125,
        feel2: int = 25,
        size1: int = 8,
        size2: int = 8,
        P1: int = 2,
        P2: int = 5,
        layer: int = 0,
        dropout2: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in ("mean", "sums", "tops"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        self.layer = layer

        feat_dim = size1 * size2
        self.eegcnn = EEGcnn(
            Chans=n_channels, dropoutRate=dropout1, kernLength1=feel1, kernLength2=feel2,
            F1=size1, D=size2, F2=feat_dim, P1=P1, P2=P2,
        )

        if layer > 0:
            self.poscode = PositionalEncoding(feat_dim, dropout=dropout2)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=max(1, feat_dim // 8),
                dim_feedforward=4 * feat_dim,
                dropout=dropout2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layer)
        else:
            self.poscode = None
            self.encoder = None

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, text_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        h = self.eegcnn(eeg).permute(0, 2, 1)       # (B, T', feat_dim)
        if self.encoder is not None:
            h = self.encoder(self.poscode(h))

        if self.pooling == "mean":
            v = h.mean(dim=1)
        elif self.pooling == "sums":
            v = h.sum(dim=1)
        else:  # "tops": first time step
            v = h[:, 0, :]

        return nn.functional.normalize(self.proj(v), dim=-1)

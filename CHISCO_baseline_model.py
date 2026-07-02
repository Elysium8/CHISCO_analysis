import math
import torch
import torch.nn as nn


class EEGcnn(nn.Module):
    """
    EEGcnn block as used in the CHISCO reference implementation.
    Expects input: (B, Chans, Samples)
    Returns:       (B, F2, Samples//(P1*P2))  ~ (B, F2, 165) for 1651 samples, P1=2, P2=5
    """
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
        dropoutType: str = "Dropout",
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, kernLength2), groups=D * F1, padding="same", bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(dropoutRate) if dropoutType == "Dropout" else nn.Dropout2d(dropoutRate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Chans, Samples)
        x = torch.unsqueeze(x, dim=1)      # (B, 1, Chans, Samples)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.squeeze(x, dim=2)        # (B, F2, Samples//(P1*P2))
        return x


class PositionalEncoding(nn.Module):
    """
    Sin/cos positional encoding in the style used in CHISCO reference code.
    Expects (B, T, D) if batch_first=True is used in the Transformer.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class CHISCOBaselineEEGEncoder(nn.Module):
    """
    Encodes EEG into a fixed-size embedding vector, following the CHISCO baseline path:
      EEGcnn -> (optional TransformerEncoder) -> pooling -> MLP projection -> normalize
    """
    def __init__(
        self,
        n_channels: int,
        text_dim: int,
        latent_dim: int = 256,
        # EEGcnn params
        dropout1: float = 0.1,
        feel1: int = 125,
        feel2: int = 25,
        size1: int = 8,   # F1
        size2: int = 8,   # D
        P1: int = 2,
        P2: int = 5,
        # Transformer params
        layer: int = 0,
        dropout2: float = 0.1,
        pooling: str = "mean",  # {"mean","sums","tops","flatten"}
    ):
        super().__init__()

        feat_dim = size1 * size2  # matches CHISCO's F2 = F1*D
        self.pooling = pooling
        self.layer = layer

        self.eegcnn = EEGcnn(
            Chans=n_channels,
            dropoutRate=dropout1,
            kernLength1=feel1,
            kernLength2=feel2,
            F1=size1,
            D=size2,
            F2=feat_dim,
            P1=P1,
            P2=P2,
        )

        if layer > 0:
            self.poscode = PositionalEncoding(feat_dim, dropout=dropout2, max_len=2000)
            nhead = max(1, feat_dim // 8)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=nhead,
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
        """
        returns: (B, text_dim) L2-normalized
        """
        # (B, feat_dim, T')
        h = self.eegcnn(eeg)
        h = h.permute(0, 2, 1)  # (B, T', feat_dim)

        if self.layer > 0:
            h = self.poscode(h)
            h = self.encoder(h)
        if self.pooling == "mean":
            v = h.mean(dim=1)  # (B, feat_dim)
        elif self.pooling == "sums":
            v = h.sum(dim=1)
        elif self.pooling == "tops":
            v = h[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        e = self.proj(v)
        return nn.functional.normalize(e, dim=-1)

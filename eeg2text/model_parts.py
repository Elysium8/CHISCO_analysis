"""Building blocks for the EEG encoder.

SpatialAttention, SubjectLayer and ResidualDilatedBlock follow the brain module
of Défossez et al. (2023), "Decoding speech perception from non-invasive brain
recordings", Nature Machine Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGFrontEnd(nn.Module):
    """EEGNet-style front end: temporal conv, depthwise spatial conv, pooling.

    (B, C, T) -> (B, F1 * D, T // 4)
    """

    def __init__(self, in_ch: int = 122, temporal_kern: int = 64, F1: int = 16, D: int = 2):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, kernel_size=(1, temporal_kern), padding=(0, temporal_kern // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(in_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                     # (B, 1, C, T)
        x = self.bn1(self.temporal(x))
        x = self.act(self.bn2(self.depthwise(x)))
        x = self.pool(x)                       # (B, F1*D, 1, T')
        return x.squeeze(2)


class SpatialAttention(nn.Module):
    """Re-mixes input channels with softmax weights predicted from sensor positions.

    (B, C_in, T) -> (B, C_out, T)
    """

    def __init__(self, in_channels: int, out_channels: int, pos_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x: torch.Tensor, sensor_pos: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_channels
        logits = self.mlp(sensor_pos).T              # (C_out, C_in)
        weights = F.softmax(logits, dim=-1)          # each output channel is a convex mix of inputs
        return torch.einsum("oc,bct->bot", weights, x)


class SubjectLayer(nn.Module):
    """A separate linear channel mixing (1x1 conv) for each subject.

    (B, C, T), subject_ids (B,) -> (B, C, T)
    """

    def __init__(self, n_subjects: int, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_subjects, channels, channels) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_subjects, channels))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        W = self.weight[subject_ids]                 # (B, C, C)
        b = self.bias[subject_ids]                   # (B, C)
        x = torch.matmul(x.transpose(1, 2), W.transpose(1, 2)) + b.unsqueeze(1)
        return x.transpose(1, 2)


class ResidualDilatedBlock(nn.Module):
    """Two residual dilated convs followed by a GLU conv (no residual)."""

    def __init__(self, channels: int, dilation1: int, dilation2: int, dilation3: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation1, dilation=dilation1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation2, dilation=dilation2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, 2 * channels, kernel_size=3, padding=dilation3, dilation=dilation3)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gelu(self.bn1(self.conv1(x)))
        x = x + self.gelu(self.bn2(self.conv2(x)))
        return F.glu(self.conv3(x), dim=1)

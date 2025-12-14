import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EEGFrontEnd(nn.Module):
    def __init__(self, in_ch=122, temporal_kern=64, F1=16, D=2):
        super().__init__()
        self.temporal = nn.Conv2d(
            1, F1, kernel_size=(1, temporal_kern), padding=(0, temporal_kern//2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(
            F1, F1*D, kernel_size=(in_ch, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.act = nn.ELU()

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)        # now (B, F1*D, 1, T')
        x = x.squeeze(2)        # (B, F1*D, T')
        return x

class SpatialAttention(nn.Module):
    """
    Simplified spatial attention over sensors.

    Args:
        in_channels:  number of EEG channels (sensors).
        out_channels: number of remapped channels after attention.
        pos_dim:      dimension of position embedding (we use 2: x,y).
        hidden_dim:   hidden size for the position MLP.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pos_dim: int = 2,
                 hidden_dim: int = 32):
        super().__init__()

        # Positions -> logits per output channel
        self.mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, sensor_pos: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, C_in, T)
        sensor_pos:  (C_in, 2) positions normalized to [-1, 1]
        returns:     (B, C_out, T)
        """
        B, C_in, T = x.shape
        assert C_in == self.in_channels, "Mismatch between input channels and sensor_pos"

        # (C_in, 2) -> (C_in, out_channels)
        logits = self.mlp(sensor_pos)  # (C_in, C_out)
        # We want attention weights over input channels for each output channel.
        # So we transpose to (C_out, C_in) and softmax over C_in:
        logits = logits.transpose(0, 1)  # (C_out, C_in)
        attn_weights = F.softmax(logits, dim=-1)  # (C_out, C_in)

        # Apply attention: for each batch, time, out_ch
        # weight matrix W: (C_out, C_in), x: (B, C_in, T) -> (B, C_out, T)
        # equivalent to x @ W^T along channels
        x_attn = torch.einsum("oc,bct->bot", attn_weights, x)  # (B, C_out, T)

        return x_attn

class SubjectLayer(nn.Module):
    """
    Subject-specific linear mapping on channels (1x1 conv per subject).

    Args:
        n_subjects:   number of subjects in the dataset.
        channels:     number of channels (after spatial attention).
    """
    def __init__(self, n_subjects: int, channels: int):
        super().__init__()
        self.n_subjects = n_subjects
        self.channels = channels

        # Weight per subject: (n_subjects, C_out, C_in); we keep C_out = C_in here
        self.weight = nn.Parameter(
            torch.randn(n_subjects, channels, channels) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(n_subjects, channels))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, C, T)
        subject_ids: (B,) integer subject indices in [0, n_subjects-1]
        returns:     (B, C, T)
        """
        B, C, T = x.shape
        assert C == self.channels, "Channel mismatch in SubjectLayer"

        # Get the weights for each sample in the batch
        W = self.weight[subject_ids]   # (B, C, C)
        b = self.bias[subject_ids]     # (B, C)

        # Apply per-sample linear map along channels
        # x: (B, C, T) -> (B, T, C)
        x_t = x.transpose(1, 2)        # (B, T, C)
        # (B, T, C) @ (B, C, C) -> (B, T, C)
        x_t = torch.matmul(x_t, W.transpose(1, 2)) + b.unsqueeze(1)
        # Back to (B, C, T)
        x = x_t.transpose(1, 2)
        return x


class ResidualDilatedBlock(nn.Module):
    """
    One residual-dilated convolutional block as described in the Nature brain module.

    Args:
        channels:   number of channels C in/out.
        dilation1:  dilation for first Conv1d.
        dilation2:  dilation for second Conv1d.
        dilation3:  dilation for third Conv1d (GLU).
    """
    def __init__(self,
                 channels: int,
                 dilation1: int,
                 dilation2: int,
                 dilation3: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            padding=dilation1,
            dilation=dilation1
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            padding=dilation2,
            dilation=dilation2
        )
        self.bn2 = nn.BatchNorm1d(channels)

        # GLU conv: double channels then GLU halves again
        self.conv3 = nn.Conv1d(
            channels, 2 * channels,
            kernel_size=3,
            padding=dilation3,
            dilation=dilation3
        )

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        # First residual conv
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = out + residual

        # Second residual conv
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        out = out + residual

        # Third conv with GLU (no residual)
        out = self.conv3(out)         # (B, 2C, T)
        out = F.glu(out, dim=1)       # (B, C, T)

        return out


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject time information.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # register_buffer ensures it's part of state_dict but not a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Sequence_Length, Batch_Size, Channels)
        """
        # Slice pe to the length of the current sequence
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
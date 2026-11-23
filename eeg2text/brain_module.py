import torch
import torch.nn as nn

from eeg2text.model_parts import ResidualDilatedBlock, SpatialAttention, SubjectLayer


class BrainModule(nn.Module):
    """Brain module adapted from Défossez et al. (2023) for EEG.

    (B, C_in, T)
      -> spatial attention over sensors        (optional)
      -> 1x1 conv to hidden_channels
      -> subject-specific layer                (optional)
      -> n_blocks residual dilated conv blocks (optional)
      -> 1x1 conv, GELU, 1x1 conv
      -> (B, out_channels, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        hidden_channels: int = 256,
        final_hidden: int = 256,
        n_blocks: int = 5,
        spatial_out_channels: int = None,
        pos_dim: int = 2,
        use_spatial_attn: bool = True,
        use_subject_layer: bool = True,
        use_residual_blocks: bool = True,
    ):
        super().__init__()
        spatial_out_channels = spatial_out_channels or in_channels

        if use_spatial_attn:
            self.spatial_attn = SpatialAttention(in_channels, spatial_out_channels, pos_dim=pos_dim, hidden_dim=32)
            in_channels = spatial_out_channels
        else:
            self.spatial_attn = None

        self.init_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.subject_layer = SubjectLayer(n_subjects, hidden_channels) if use_subject_layer else None

        n_blocks = n_blocks if use_residual_blocks else 0
        self.blocks = nn.ModuleList(
            ResidualDilatedBlock(hidden_channels, dilation1=2 ** i, dilation2=2 ** (i + 1), dilation3=2 ** (i + 1))
            for i in range(n_blocks)
        )

        self.final_conv1 = nn.Conv1d(hidden_channels, final_hidden, kernel_size=1)
        self.final_gelu = nn.GELU()
        self.final_conv2 = nn.Conv1d(final_hidden, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, sensor_pos: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        if self.spatial_attn is not None:
            x = self.spatial_attn(x, sensor_pos)
        x = self.init_conv(x)
        if self.subject_layer is not None:
            x = self.subject_layer(x, subject_ids)
        for block in self.blocks:
            x = block(x)
        return self.final_conv2(self.final_gelu(self.final_conv1(x)))

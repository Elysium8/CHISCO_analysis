import torch
import torch.nn as nn
import torch.nn.functional as F

from model_parts import SpatialAttention, SubjectLayer, ResidualDilatedBlock


class BrainModule(nn.Module):
    """
    Nature-style brain module adapted for EEG/CHISCO.

    Pipeline:
        X (B, C_in, T)
          -> spatial attention (with sensor positions)
          -> Conv1x1 (channel remap)
          -> subject-specific layer
          -> N residual dilated conv blocks
          -> Conv1x1 + GELU + Conv1x1
          -> Z (B, out_channels, T)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_subjects: int,
                 hidden_channels: int = 256,
                 final_hidden: int = 256,
                 n_blocks: int = 5,
                 spatial_out_channels: int = None,
                 pos_dim: int = 2):
        super().__init__()

        if spatial_out_channels is None:
            spatial_out_channels = in_channels  # can choose larger if you like

        # 1) Spatial attention over sensors
        self.spatial_attn = SpatialAttention(
            in_channels=in_channels,
            out_channels=spatial_out_channels,
            pos_dim=pos_dim,
            hidden_dim=32
        )

        # Initial 1x1 conv (no activation in paper, but we can keep it linear here)
        self.init_conv = nn.Conv1d(
            spatial_out_channels,
            hidden_channels,
            kernel_size=1
        )

        # 2) Subject-specific 1x1 conv
        self.subject_layer = SubjectLayer(
            n_subjects=n_subjects,
            channels=hidden_channels
        )

        # 3) Residual dilated conv blocks
        blocks = []
        for i in range(n_blocks):
            # Example dilation schedule (you can tweak this):
            d1 = 2 ** i
            d2 = 2 ** (i + 1)
            d3 = 2 ** (i + 1)
            blocks.append(
                ResidualDilatedBlock(
                    channels=hidden_channels,
                    dilation1=d1,
                    dilation2=d2,
                    dilation3=d3
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # 4) Final 1x1 convs to get brain latent representation
        self.final_conv1 = nn.Conv1d(
            hidden_channels,
            final_hidden,
            kernel_size=1
        )
        self.final_gelu = nn.GELU()
        self.final_conv2 = nn.Conv1d(
            final_hidden,
            out_channels,
            kernel_size=1
        )

    def forward(self,
                x: torch.Tensor,
                sensor_pos: torch.Tensor,
                subject_ids: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, C_in, T)       EEG
        sensor_pos:  (C_in, 2)          sensor positions
        subject_ids: (B,)               subject indices

        returns:
            z: (B, out_channels, T)
        """
        # 1. Spatial attention
        x = self.spatial_attn(x, sensor_pos)  # (B, spatial_out_channels, T)

        # Initial 1x1 conv
        x = self.init_conv(x)                 # (B, hidden_channels, T)

        # 2. Subject layer
        x = self.subject_layer(x, subject_ids)  # (B, hidden_channels, T)

        # 3. Residual dilated blocks
        for block in self.blocks:
            x = block(x)                      # (B, hidden_channels, T)

        # 4. Final 1x1 convs
        x = self.final_conv1(x)               # (B, final_hidden, T)
        x = self.final_gelu(x)
        z = self.final_conv2(x)               # (B, out_channels, T)

        return z

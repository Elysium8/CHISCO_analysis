# Brain_Module.py
import torch
import torch.nn as nn

from model_parts import SpatialAttention, SubjectLayer, ResidualDilatedBlock


class BrainModule(nn.Module):
    """
    Nature-style brain module adapted for EEG/CHISCO.

    Pipeline:
        X (B, C_in, T)
          -> (optional) spatial attention (with sensor positions)
          -> Conv1x1 (channel remap)
          -> (optional) subject-specific layer
          -> (optional) N residual dilated conv blocks
          -> Conv1x1 + GELU + Conv1x1
          -> Z (B, out_channels, T)
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
        # toggles
        use_spatial_attn: bool = True,
        use_subject_layer: bool = True,
        use_residual_blocks: bool = True,
    ):
        super().__init__()

        self.use_spatial_attn = use_spatial_attn
        self.use_subject_layer = use_subject_layer
        self.use_residual_blocks = use_residual_blocks

        if spatial_out_channels is None:
            spatial_out_channels = in_channels

        # 1) Spatial attention over sensors (optional)
        if self.use_spatial_attn:
            self.spatial_attn = SpatialAttention(
                in_channels=in_channels,
                out_channels=spatial_out_channels,
                pos_dim=pos_dim,
                hidden_dim=32,
            )
            attn_out_channels = spatial_out_channels
        else:
            self.spatial_attn = None
            attn_out_channels = in_channels

        # Initial 1x1 conv
        self.init_conv = nn.Conv1d(attn_out_channels, hidden_channels, kernel_size=1)

        # 2) Subject-specific layer (optional)
        if self.use_subject_layer:
            self.subject_layer = SubjectLayer(n_subjects=n_subjects, channels=hidden_channels)
        else:
            self.subject_layer = None

        # 3) Residual dilated conv blocks (optional)
        if self.use_residual_blocks:
            blocks = []
            for i in range(n_blocks):
                d1 = 2 ** i
                d2 = 2 ** (i + 1)
                d3 = 2 ** (i + 1)
                blocks.append(
                    ResidualDilatedBlock(
                        channels=hidden_channels,
                        dilation1=d1,
                        dilation2=d2,
                        dilation3=d3,
                    )
                )
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = nn.ModuleList([])

        # 4) Final 1x1 convs
        self.final_conv1 = nn.Conv1d(hidden_channels, final_hidden, kernel_size=1)
        self.final_gelu = nn.GELU()
        self.final_conv2 = nn.Conv1d(final_hidden, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, sensor_pos: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        # 1) Spatial attention (optional)
        if self.use_spatial_attn:
            x = self.spatial_attn(x, sensor_pos)

        # Initial 1x1 conv
        x = self.init_conv(x)

        # 2) Subject layer (optional)
        if self.use_subject_layer:
            x = self.subject_layer(x, subject_ids)

        # 3) Residual dilated blocks (optional)
        if self.use_residual_blocks:
            for block in self.blocks:
                x = block(x)

        # 4) Final 1x1 convs
        x = self.final_conv1(x)
        x = self.final_gelu(x)
        z = self.final_conv2(x)
        return z

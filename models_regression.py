import torch
import torch.nn as nn
import torch.nn.functional as F

from Brain_Module import BrainModule
from model_parts import EEGFrontEnd

class EEGToSentenceLengthModel(nn.Module):
    """
    EEGFrontEnd -> BrainModule -> attention pooling -> MLP -> scalar length (log space).
    """
    def __init__(
        self,
        n_channels: int,
        n_subjects: int,
        latent_dim: int = 256,
        hidden_channels: int = 256,
        final_hidden: int = 256,
        n_blocks: int = 5,
        attn_hidden: int = 128,
        use_subject_embed_in_head: bool = True,
    ):
        super().__init__()

        self.eeg_frontend = EEGFrontEnd(
            in_ch=n_channels,
            temporal_kern=64,
            F1=16,
            D=2,
        )
        frontend_out_ch = 16 * 2

        self.brain = BrainModule(
            in_channels=frontend_out_ch,
            out_channels=latent_dim,
            n_subjects=n_subjects,
            hidden_channels=hidden_channels,
            final_hidden=final_hidden,
            n_blocks=n_blocks,
            spatial_out_channels=frontend_out_ch,
        )

        self.sensor_pos = nn.Parameter(torch.randn(frontend_out_ch, 2) * 0.1)

        # Attention pooling over time: z (B,D,T') -> weights (B,T')
        self.attn = nn.Sequential(
            nn.Conv1d(latent_dim, attn_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(attn_hidden, 1, kernel_size=1),
        )

        # Optional subject embedding in the final head (helps calibration)
        self.use_subject_embed_in_head = use_subject_embed_in_head
        if use_subject_embed_in_head:
            self.subj_emb = nn.Embedding(n_subjects, 32)
            head_in = latent_dim + 32
        else:
            head_in = latent_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: y_hat_log (B,) predicted log1p(length_chars)
        """
        x = self.eeg_frontend(x)  # (B, C', T')
        z = self.brain(x, self.sensor_pos, subject_ids)  # (B, D, T')

        # Attention weights over time
        w_logits = self.attn(z).squeeze(1)          # (B, T')
        w = torch.softmax(w_logits, dim=-1)         # (B, T')

        # Weighted sum pooling
        z_pool = torch.einsum("bdt,bt->bd", z, w)   # (B, D)

        if self.use_subject_embed_in_head:
            s = self.subj_emb(subject_ids)          # (B, 32)
            z_pool = torch.cat([z_pool, s], dim=-1)

        y_hat = self.head(z_pool).squeeze(-1)       # (B,)
        return y_hat

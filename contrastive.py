import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    """
    Wraps the standard ChiscoImagineDataset to return TWO augmented views
    of the same EEG trial for SimCLR training.
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # We assume base_dataset[idx] returns a dict {'eeg': ..., 'subject_id': ...}
        # We need to manually unpack or access the raw tensor.
        # Based on your previous code structure:

        raw_eeg = self.base_dataset.eeg[idx]  # (C, T) tensor
        subj_id = self.base_dataset.subject_ids[idx]

        # Apply augmentations twice to get two views
        # Note: We clone to ensure they are distinct in memory
        view_1 = self.transform(raw_eeg.clone())
        view_2 = self.transform(raw_eeg.clone())

        return {
            "view1": view_1,
            "view2": view_2,
            "subject_id": subj_id
        }


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for SimCLR.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (B, D) latent representations from the two views
        """
        batch_size = z_i.shape[0]

        # Concatenate: (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix (cosine similarity)
        # z needs to be normalized already, or we norm here:
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)

        # Positive pairs: (i, i+B) and (i+B, i)
        # We construct targets for CrossEntropy
        # For index i (0 to B-1), the positive match is i + B
        # For index i + B (B to 2B-1), the positive match is i
        target = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        loss = F.cross_entropy(sim_matrix, target)
        return loss


class SimCLR(nn.Module):
    """
    Wrapper for BrainModule that adds a Projection Head.
    The projection head is used ONLY during pre-training.
    """

    def __init__(self, brain_encoder, latent_dim=256, proj_dim=128):
        super().__init__()
        self.encoder = brain_encoder

        # Projector: MLP with one hidden layer
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )

    def forward(self, x, sensor_pos, subject_ids):
        # Get representation from BrainModule
        # Note: BrainModule returns (B, out_channels, T)
        # We need to pool it first (SimCLR works on vectors)

        # Pass through BrainModule
        features_seq = self.encoder(x, sensor_pos, subject_ids)

        # Global Mean Pooling over time
        h = features_seq.mean(dim=-1)

        # Project
        z = self.projector(h)
        return h, z
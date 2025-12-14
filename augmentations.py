import torch
import torch.nn as nn
import random


class EEGAugment(nn.Module):
    def __init__(self,
                 p_dropout=0.5,
                 max_channel_drop=2,
                 noise_std=0.01,
                 time_shift_max=10):
        super().__init__()
        self.p_dropout = p_dropout
        self.max_channel_drop = max_channel_drop
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max

    def forward(self, x):
        """
        x: (Channels, Time)
        """
        # 1. Gaussian Noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 2. Channel Dropout (Spatial corruption)
        # Randomly zero out 1 to N channels
        if random.random() < self.p_dropout:
            n_channels = x.shape[0]
            # Pick how many channels to drop (1 to max_channel_drop)
            n_drop = random.randint(1, min(self.max_channel_drop, n_channels))
            drop_indices = random.sample(range(n_channels), n_drop)
            x[drop_indices, :] = 0.0

        # 3. Time Shift
        # Roll the signal slightly left or right
        if self.time_shift_max > 0:
            shift = random.randint(-self.time_shift_max, self.time_shift_max)
            x = torch.roll(x, shifts=shift, dims=-1)

            # Zero-pad the gap created by roll if you don't want circular shift
            # (optional: here we assume circular is okay or small enough)

        return x
"""Shallow Conv1D feature extractor, shared by both input branches.

``SequenceConvEncoder`` turns a per-day feature sequence ``[.., T, F]`` into per-day
embeddings ``[.., T, C]`` (same length, so the LSTM still sees one vector per
lookback day). Both branches use it: one-to-two Conv1D layers with GELU, norm,
and dropout.
"""

from __future__ import annotations

import torch
from torch import nn


class SequenceConvEncoder(nn.Module):
    """[*, T, F] -> [*, T, C] per-day embeddings (length preserved)."""

    def __init__(self, in_features: int, channels: int, kernel: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel, padding=pad),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad),
            nn.GELU(),
        )
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-2]
        t, f = x.shape[-2], x.shape[-1]
        x = x.reshape(-1, t, f).transpose(1, 2)  # [B, F, T]
        x = self.net(x).transpose(1, 2)  # [B, T, C]
        return x.reshape(*lead, t, self.out_dim)

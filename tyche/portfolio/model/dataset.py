"""Torch dataset over windowed cross-sectional samples.

Each item is one decision day: synchronized ``[N, T, ...]`` lookback tensors for both
branches plus the ``[N]`` forward-return target. Tensors are sliced lazily from
the (already standardized) aligned arrays, so memory stays flat regardless of sample
count.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from tyche.portfolio.data.assemble import AlignedData
from tyche.portfolio.data.windows import Sample


class WindowDataset(Dataset):
    def __init__(self, data: AlignedData, samples: list[Sample]):
        self.data = data
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = self.samples[i]
        sl = s.day_slice
        return {
            "daily": torch.from_numpy(np.ascontiguousarray(self.data.daily[:, sl])),
            "news": torch.from_numpy(np.ascontiguousarray(self.data.news[:, sl])),
            "target": torch.from_numpy(s.target),
            "t": torch.tensor(s.t),
        }

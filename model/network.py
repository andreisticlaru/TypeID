"""LSTM encoder: keystroke timing sequence -> 128-dim embedding.

Architecture per ARCHITECTURE.md:
    Input (batch, M=50, 4) -> LSTM(128) + dropout -> LSTM(128) + dropout
    -> masked mean-pool over timesteps -> Dense(128) -> L2 normalize

Padding is always at the end of a window (real keystrokes come first), and
the LSTM is unidirectional, so an output at a real timestep never depends on
the padded timesteps after it. That's what makes masked mean-pooling correct
without needing `pack_padded_sequence`: we simply exclude padded positions
from the average after the fact, rather than preventing the LSTM from seeing
them in the first place.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 4
HIDDEN_SIZE = 128
EMBEDDING_DIM = 128
DROPOUT = 0.2


class KeystrokeEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, M, input_size) float
        mask: (batch, M) bool, True where x holds real (non-padded) data
        Returns: (batch, embedding_dim), L2-normalized
        """
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        mask_f = mask.unsqueeze(-1).float()  # (batch, M, 1)
        summed = (out * mask_f).sum(dim=1)  # (batch, hidden_size)
        counts = mask_f.sum(dim=1).clamp(min=1)  # (batch, 1), avoid div-by-zero
        pooled = summed / counts

        embedding = self.fc(pooled)
        return F.normalize(embedding, p=2, dim=1)

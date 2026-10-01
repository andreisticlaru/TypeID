"""TypeNet (Acien et al. 2021, arXiv 2101.05570, Sec 4.2) rebuilt in PyTorch, timing features only.

    Input (batch, M=50, 4) in ms -> /1000 (the paper feeds seconds)
    -> MaskedBatchNorm(4)
    -> LSTM(128), recurrent dropout 0.2
    -> MaskedBatchNorm(128) -> Dropout(0.5)
    -> LSTM(128), recurrent dropout 0.2
    -> hidden state at the last real keystroke = the 128-d embedding (no Linear, no normalization)

The paper's 5th input, the key code, is left out on purpose: this project never shows the model which
key was pressed (see CLAUDE.md, "Letting the network memorize content").

Why a hand-written LSTM instead of nn.LSTM: Keras' recurrent_dropout drops units of h_{t-1} inside the
recurrence, with one mask per sequence reused at every timestep (Gal & Ghahramani). nn.LSTM only
offers dropout between layers, so matching the paper needs the loop. Keras initialisation is copied
too (glorot-uniform input weights, orthogonal recurrent weights, forget-gate bias 1), and there is a
single bias vector per layer, as in Keras.

Masking: Keras' Masking layer makes the LSTM skip padded steps and return the output at the last real
step. Padding here is always at the end and the LSTM is unidirectional, so running over the padding
and reading h at index (length - 1) gives exactly that. BatchNorm statistics are computed over real
timesteps only, so the padded zeros can't shift the normalisation of real keystrokes.

Ablation flags (defaults = the paper), each borrowed from our v1 encoder so one ingredient can be tested at a
time: normalize=True L2-normalizes the output (for cosine distance); readout="mean" averages h over the real
steps instead of taking the last one; batchnorm=False removes both MaskedBatchNorms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 4
HIDDEN_SIZE = 128
DROPOUT = 0.5
RECURRENT_DROPOUT = 0.2
INPUT_SCALE = 1000.0  # ms -> s


class MaskedBatchNorm(nn.Module):
    """BatchNorm over the feature axis of (batch, time, C), statistics from mask==True steps only.

    Keras defaults: momentum 0.99 on the running average (= PyTorch's 0.01 update weight), eps 1e-3.
    """

    def __init__(self, num_features: int, momentum: float = 0.01, eps: float = 1e-3):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, groups: int = 1) -> torch.Tensor:
        """`groups` > 1 splits the batch into equal consecutive chunks with their own statistics, which
        equals running the chunks through separate calls (how Keras treats a shared siamese branch)."""
        if not self.training:
            return (x - self.running_mean) * torch.rsqrt(self.running_var + self.eps) * self.weight + self.bias
        shape = x.shape
        xg = x.reshape(groups, -1, *shape[1:])  # (groups, B/groups, T, C)
        mg = mask.reshape(groups, -1, shape[1], 1).to(x.dtype)
        count = mg.sum((1, 2), keepdim=True)
        mean = (xg * mg).sum((1, 2), keepdim=True) / count
        var = ((xg - mean).pow(2) * mg).sum((1, 2), keepdim=True) / count
        with torch.no_grad():
            for g in range(groups):  # one running-average update per branch call, in order
                self.running_mean.lerp_(mean[g].flatten(), self.momentum)
                self.running_var.lerp_(var[g].flatten(), self.momentum)
        out = (xg - mean) * torch.rsqrt(var + self.eps)
        return out.reshape(shape) * self.weight + self.bias


class RecurrentDropoutLSTM(nn.Module):
    """Single-layer LSTM with Keras-style recurrent dropout; returns every timestep's h."""

    def __init__(self, input_size: int, hidden_size: int, recurrent_dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout
        # Gate order i, f, g, o (same as Keras' i, f, c, o).
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        for gate in self.weight_hh.data.chunk(4):  # Keras orthogonalises the whole (H, 4H) kernel; per-gate is the same distribution
            nn.init.orthogonal_(gate)
        with torch.no_grad():
            self.bias[hidden_size:2 * hidden_size] = 1.0  # unit_forget_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        gates_x = x @ self.weight_ih.T + self.bias  # input projection for all steps at once
        drop = None
        if self.training and self.recurrent_dropout > 0:
            keep = 1 - self.recurrent_dropout
            drop = torch.bernoulli(x.new_full((batch, self.hidden_size), keep)) / keep
        outputs = []
        # unbind, not gates_x[:, t]: each slice's backward would allocate a full-size zero gradient.
        for gx in gates_x.unbind(1):
            h_in = h * drop if drop is not None else h
            i, f, g, o = (gx + h_in @ self.weight_hh.T).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class TypeNetEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        dropout: float = DROPOUT,
        recurrent_dropout: float = RECURRENT_DROPOUT,
        input_scale: float = INPUT_SCALE,
        normalize: bool = False,
        readout: str = "last",
        batchnorm: bool = True,
    ):
        super().__init__()
        if readout not in ("last", "mean"):
            raise ValueError(f"readout must be 'last' or 'mean', got {readout!r}")
        self.input_scale = input_scale
        self.normalize, self.readout, self.batchnorm = normalize, readout, batchnorm
        # Created in this order so the default model draws the same initial weights as before the flags existed.
        self.bn1 = MaskedBatchNorm(input_size) if batchnorm else None
        self.lstm1 = RecurrentDropoutLSTM(input_size, hidden_size, recurrent_dropout)
        self.bn2 = MaskedBatchNorm(hidden_size) if batchnorm else None
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = RecurrentDropoutLSTM(hidden_size, hidden_size, recurrent_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, groups: int = 1) -> torch.Tensor:
        """x: (batch, M, 4) ms; mask: (batch, M) bool. Returns (batch, hidden) raw embeddings.

        groups: see MaskedBatchNorm. Training passes anchor, positive and negative as one batch with
        groups=3; that's ~3x faster than three calls (the time-step loop dominates) and gives the same result.
        """
        out = x / self.input_scale
        if self.batchnorm:
            out = self.bn1(out, mask, groups)
        out = self.lstm1(out)
        if self.batchnorm:
            out = self.bn2(out, mask, groups)
        out = self.lstm2(self.dropout(out))
        if self.readout == "last":
            last = (mask.sum(1) - 1).clamp(min=0)
            out = out[torch.arange(len(out), device=out.device), last]
        else:
            m = mask.unsqueeze(-1).to(out.dtype)
            out = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(out, dim=1) if self.normalize else out

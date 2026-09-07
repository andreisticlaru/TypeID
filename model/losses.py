"""Triplet loss for the keystroke embedding network.

Formula (ARCHITECTURE.md): L = max(0, d(A, P) - d(A, N) + margin)

Distance metric: cosine distance. KeystrokeEncoder already L2-normalizes its
output (see model/network.py), so for two unit-norm embeddings, cosine
similarity is just their dot product -- no need to divide by norms again.
Cosine distance = 1 - cosine similarity.
"""

from __future__ import annotations

import torch

MARGIN = 0.5


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, for a batch of L2-normalized embedding pairs.

    a, b: (batch, embedding_dim), each row already unit-norm.
    Returns: (batch,) distance per pair.
    """
    # TODO: cosine similarity between two unit vectors is just their dot
    # product -- elementwise multiply then sum over the embedding dim
    # (dim=1). Return 1 - that.
    raise NotImplementedError


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = MARGIN,
) -> torch.Tensor:
    """Triplet margin loss, averaged over the batch.

    anchor, positive, negative: (batch, embedding_dim) L2-normalized
    embeddings -- three separate forward passes of the same KeystrokeEncoder.
    Returns: scalar loss.
    """
    # TODO:
    #   1. d_ap = cosine_distance(anchor, positive)
    #   2. d_an = cosine_distance(anchor, negative)
    #   3. per_example = max(0, d_ap - d_an + margin)  -- torch.clamp(..., min=0)
    #      or torch.nn.functional.relu(...)
    #   4. return per_example.mean()
    raise NotImplementedError

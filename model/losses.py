"""Triplet loss for the keystroke embedding network.

Triplet construction:
- Anchor (A): a keystroke sequence from person X, typed at session/sentence S1
- Positive (P): a different keystroke sequence from the SAME person X, but different
  session/sentence (S2 ≠ S1). Forces the model to learn person-specific rhythm,
  not sentence-specific timing.
- Negative (N): a keystroke sequence from a DIFFERENT person Y (Y ≠ X).

Loss formula: L = max(0, d(A, P) - d(A, N) + margin)

The loss pushes anchor and positive embeddings close together (minimize d(A, P))
and anchor and negative far apart (maximize d(A, N)), with a safety margin. The
max(0, ...) means if negative is already far enough (d(A, N) > d(A, P) + margin),
the loss is zero and gradients don't flow — we've already won.

**Distance metric:** cosine distance. KeystrokeEncoder already L2-normalizes its
output (see model/network.py), so for two unit-norm embeddings, cosine
similarity is just their dot product. Cosine distance = 1 - cosine similarity.
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


    return a.mul(b).sum(dim=1).neg().add(1)



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
   
    d_ap = cosine_distance(anchor, positive)
    d_an = cosine_distance(anchor, negative)
    per_example = torch.clamp(d_ap - d_an + margin, min=0)
    return per_example.mean()

    

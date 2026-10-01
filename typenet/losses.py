"""TypeNet's triplet loss (Eq. 4): max(0, ||a-p||^2 - ||a-n||^2 + alpha), alpha = 1.5, averaged.

Unlike model/losses.py (cosine distance on unit vectors, margin 0.5), TypeNet's embeddings are not
normalised, so the model can also grow or shrink distances by scaling its outputs.

distance="cosine" is model/losses.py's rule (1 - a.b, valid only on unit vectors), for the ablation that gives
TypeNet an L2-normalized output; the caller passes the matching margin.
"""

from __future__ import annotations

import torch

MARGIN = 1.5


def squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).sum(1)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - (a * b).sum(1)


def pair_distance(a, b, distance: str) -> torch.Tensor:
    return squared_distance(a, b) if distance == "sqeuclidean" else cosine_distance(a, b)


def triplet_loss(anchor, positive, negative, margin: float = MARGIN, distance: str = "sqeuclidean") -> torch.Tensor:
    return torch.clamp(pair_distance(anchor, positive, distance) - pair_distance(anchor, negative, distance) + margin,
                       min=0).mean()


def mined_loss(anchor, positive, negative, same_subject, mode: str, margin: float = MARGIN,
               distance: str = "sqeuclidean") -> torch.Tensor:
    """Triplet loss with each anchor's negative swapped for a harder one already in the batch.

    Same candidate pool (every positive and negative, 2B), same-subject mask and selection rules as
    model/train.py's mine_hardest / mine_semi_hard, so the only difference from our encoder's mining
    is the distance:
      "hard": the closest valid candidate, even one already closer than the positive.
      "semi": the closest candidate still farther than the positive; if none, the farthest valid one.
    """
    pool = torch.cat([positive, negative])
    if distance == "sqeuclidean":
        # ||a||^2 + ||n||^2 - 2 a.n rather than cdist(...)**2: cdist's gradient is NaN at zero distance.
        d_an = anchor.pow(2).sum(1, keepdim=True) + pool.pow(2).sum(1) - 2 * anchor @ pool.T
    else:
        d_an = 1 - anchor @ pool.T
    d_ap = pair_distance(anchor, positive, distance)
    valid = d_an.detach().masked_fill(same_subject, float("inf"))
    if mode == "hard":
        idx = valid.argmin(1)
    else:
        semi = valid.masked_fill(valid <= d_ap.detach()[:, None], float("inf"))
        idx = semi.argmin(1)
        none_qualify = semi.gather(1, idx[:, None]).isinf().squeeze(1)
        farthest = valid.masked_fill(valid.isinf(), -float("inf")).argmax(1)
        idx = torch.where(none_qualify, farthest, idx)
    return torch.clamp(d_ap - d_an.gather(1, idx[:, None]).squeeze(1) + margin, min=0).mean()

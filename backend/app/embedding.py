"""The frozen embedding model f(): extracted timing windows -> one 128-dim L2-normalized vector.

Everything downstream (enrollment, identification) is a forward pass through this
plus vector comparison. The checkpoint is loaded once, on first use.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from model.network import KeystrokeEncoder

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "model" / "encoder_hard.pt"


@lru_cache(maxsize=1)
def _load_encoder() -> KeystrokeEncoder:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    encoder = KeystrokeEncoder(**ckpt["config"]["model"])
    encoder.load_state_dict(ckpt["state_dict"])
    return encoder.eval()  # eval(): dropout off


@torch.no_grad()
def embed(features: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Embed one typing session.

    `features` is the (windows, mask) pair from ``features.extract.extract_features``.
    A session longer than M keystrokes yields several windows; their embeddings are
    averaged and re-normalized (the same pooling eval/rank_n.py measured), so dot
    product stays cosine similarity.
    """
    windows, mask = features
    window_embeddings = _load_encoder()(torch.from_numpy(windows), torch.from_numpy(mask)).numpy()
    mean = window_embeddings.mean(axis=0)
    return mean / np.linalg.norm(mean)

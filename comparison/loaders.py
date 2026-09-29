"""Load any trained encoder from its checkpoint, whichever architecture produced it."""

from __future__ import annotations

import torch

from model.network import KeystrokeEncoder
from typenet.network import TypeNetEncoder

# Checkpoints from model/train.py predate the "arch" key; they are all KeystrokeEncoder.
ARCHITECTURES = {"v1": KeystrokeEncoder, "typenet": TypeNetEncoder}


def load_any_encoder(path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    model = ARCHITECTURES[config.get("arch", "v1")](**config["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()

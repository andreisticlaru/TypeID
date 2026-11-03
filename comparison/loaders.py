"""Load any trained encoder from its checkpoint, whichever architecture produced it."""

from __future__ import annotations

import torch

from model.network import KeystrokeEncoder
from typenet.network import TypeNetEncoder

# Checkpoints from model/train.py predate the "arch" key; they are all KeystrokeEncoder.
ARCHITECTURES = {"v1": KeystrokeEncoder, "typenet": TypeNetEncoder}


IL = 1  # (HL, IL, PL, RL)
AALTO_IL_P99 = 1113.0  # ms, from 200k windows of data/preprocessed_typenet


class PauseEdit(torch.nn.Module):
    """Scoring-only input edit, to test whether long pauses carry the task gap. The extractor stays untouched.

    A pause is one stretch of time that shows up in IL, PL and RL together, so all three shrink by the same amount:
    the input stays one a keyboard could have produced, and HL is never changed.
      cap:   IL above AALTO_IL_P99 becomes exactly that
      soft:  the excess is log-compressed instead, so a 60 s pause still reads longer than a 2 s one
      noise: control, the amounts cap would remove are moved to random real steps with random signs, so the
             same quantity of timing changes but not specifically at long pauses
    """

    def __init__(self, encoder: torch.nn.Module, mode: str, cap: float = AALTO_IL_P99):
        super().__init__()
        self.encoder, self.mode, self.cap = encoder, mode, cap

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        excess = (x[..., IL] - self.cap).clamp(min=0)
        if self.mode == "soft":
            delta = excess - self.cap * torch.log1p(excess / self.cap)
        elif self.mode == "cap":
            delta = excess
        else:
            # padding sorts last, and excess sits on real steps only, so this shuffles within the real steps
            order = (torch.rand_like(excess) + (~mask).float()).argsort(dim=1)
            delta = excess.gather(1, order) * (2 * torch.randint_like(excess, 2) - 1) * mask
        return self.encoder(torch.cat([x[..., :IL], x[..., IL:] - delta[..., None]], dim=-1), mask)


def load_any_encoder(path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    model = ARCHITECTURES[config.get("arch", "v1")](**config["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()

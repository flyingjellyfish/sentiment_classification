"""Aligned-50 validity, source-zero and injected-span masks (L,V,A order)."""
from __future__ import annotations

import numpy as np
import torch


NAMES_LVA = ("text", "vision", "audio")
SCENARIOS = {"text": (0,), "vision": (1,), "audio": (2,),
             "audio_vision": (1, 2), "all_three": (0, 1, 2)}


def mask_state(audio: torch.Tensor, vision: torch.Tensor, lengths: torch.Tensor,
               *, mark_source_zeros: bool = True):
    """Return content, source_zero, observed as B×3×50 boolean tensors.

    Padding is excluded by ``content`` but remains observed in the model input
    mask so that the inherited missing-token mechanism does not replace it.
    Optional zero_padding in EMOE.forward clears padded features separately.
    """
    if audio.ndim != 3 or vision.ndim != 3 or audio.shape[:2] != vision.shape[:2]:
        raise ValueError("audio and vision must be B×50×D with matching B,50")
    if audio.shape[1] != 50 or tuple(lengths.shape) != (audio.shape[0],):
        raise ValueError("aligned_50 expects B×50 inputs and B lengths")
    if not bool(((lengths >= 3) & (lengths <= 50)).all()):
        raise ValueError("text_bert effective length must be 3..50")
    position = torch.arange(50, device=audio.device)[None, :]
    content_2d = (position >= 1) & (position < lengths[:, None] - 1)
    content = content_2d[:, None, :].expand(-1, 3, -1)
    source_zero = torch.zeros_like(content)
    source_zero[:, 1] = (vision == 0).all(dim=2) & content_2d
    source_zero[:, 2] = (audio == 0).all(dim=2) & content_2d
    observed = torch.ones_like(content)
    if mark_source_zeros:
        observed &= ~source_zero
    return content, source_zero, observed


def inject_spans(observed: torch.Tensor, content: torch.Tensor, lengths: torch.Tensor,
                 rng: np.random.Generator, *, kind: str = "random",
                 position: str = "random", fraction: float | None = None,
                 chance: float = 0.8, spans: int = 1,
                 include_all_three: bool = False):
    """Inject 1–2 contiguous spans, and return the new mask and *new* deletions."""
    if observed.shape != content.shape or observed.shape[1:] != (3, 50):
        raise ValueError("LVA masks must be B×3×50")
    if spans not in (1, 2):
        raise ValueError("spans must be one or two")
    if kind != "random" and kind not in SCENARIOS:
        raise ValueError(f"unknown modality scenario {kind}")
    if position not in ("random", "start", "middle", "end"):
        raise ValueError(f"unknown position {position}")
    result = observed.clone()
    for i in range(len(result)):
        if rng.random() > chance:
            continue
        n = int(lengths[i]) - 2
        if n < 2:
            continue
        rate = float(rng.uniform(0.15, 0.5) if fraction is None else fraction)
        if not (0 < rate < 1):
            raise ValueError("fraction must be in (0,1)")
        total = max(1, min(n - 1, round(rate * n)))
        if kind == "random":
            choices = ((0,), (1,), (2,), (1, 2), (0, 1), (0, 2))
            if include_all_three:
                choices += ((0, 1, 2),)
            channels = choices[int(rng.integers(len(choices)))]
        else:
            channels = SCENARIOS[kind]
        widths = [total] if spans == 1 or total < 2 else [total // 2, total - total // 2]
        for j, width in enumerate(widths):
            if position == "start":
                start = 1 + sum(widths[:j])
            elif position == "end":
                start = 1 + n - sum(widths[j:])
            elif position == "middle":
                start = 1 + (n - total) // 2 + sum(widths[:j])
            else:
                start = 1 + int(rng.integers(0, n - width + 1))
            result[i, channels, start:start + width] = False
    injected = observed & ~result & content
    return result, injected


def deletion_stats(base: torch.Tensor, injected: torch.Tensor, content: torch.Tensor):
    denom = content.float().sum(dim=2).clamp_min(1)
    actual_fraction = injected.float().sum(dim=2) / denom
    return {"actual_fraction_LVA": actual_fraction.mean(dim=0).tolist(),
            "no_new_deletion_LVA": (injected.sum(dim=2) == 0).sum(dim=0).tolist(),
            "source_unavailable_LVA": ((~base & content).sum(dim=2) > 0).sum(dim=0).tolist()}

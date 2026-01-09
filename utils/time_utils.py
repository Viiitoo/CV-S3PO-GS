# utils/time_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

Number = Union[int, float]

def normalize_frame_time(frame_idx: Number,
                         num_frames: Optional[int] = None,
                         *,
                         clamp: bool = True,
                         eps: float = 1e-8) -> float:
    """
    EH-style normalized timestamp for each frame.
    - If num_frames is provided: t = idx / (num_frames - 1)
    - Else: fallback to idx (NOT recommended, but avoids crash)

    Returns:
        float in [0,1] if clamp=True and num_frames is provided.
    """
    idx = float(frame_idx)

    if num_frames is None:
        # fallback: you can still run, but time scale will be off
        t = idx
    else:
        denom = max(eps, float(num_frames - 1))
        t = idx / denom

    if clamp:
        if t < 0.0: t = 0.0
        if t > 1.0: t = 1.0
    return float(t)


def attach_time_to_viewpoint(viewpoint_camera,
                             frame_idx: Number,
                             num_frames: Optional[int] = None,
                             *,
                             attr: str = "time") -> float:
    """
    Compute EH-style time and attach it to viewpoint_camera.<attr>.
    Returns t.
    """
    t = normalize_frame_time(frame_idx, num_frames)
    setattr(viewpoint_camera, attr, t)
    return t

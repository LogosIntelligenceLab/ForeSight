"""Continuous-time features for video and trajectory tokens."""

import math

import torch
from torch import nn


def future_timestamps(count, frequency, *, device=None):
    if count < 1 or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Frame count and sampling frequency must be positive")
    return torch.arange(1, count + 1, device=device, dtype=torch.float32) / frequency


def expand_timestamps(times, batch, length, device, valid=None):
    times = torch.as_tensor(times, device=device, dtype=torch.float32)
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(batch, -1)
    if times.shape != (batch, length):
        raise ValueError(
            f"Expected timestamps [{batch}, {length}], got {tuple(times.shape)}"
        )
    valid = torch.ones_like(times, dtype=torch.bool) if valid is None else valid
    if not torch.isfinite(times[valid]).all():
        raise ValueError("Timestamps must be finite")
    for row, mask in zip(times, valid):
        selected = row[mask]
        if selected.numel() > 1 and not (selected[1:] > selected[:-1]).all():
            raise ValueError("Frame timestamps must increase strictly")
    return times.masked_fill(~valid, 0)


class TimeEncoding(nn.Module):
    """Sinusoidal encoding of seconds relative to the current observation."""

    def __init__(self, dim):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("Time encoding dimension must be positive and even")
        frequencies = 2 * math.pi * 10000 ** (-torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, seconds):
        angles = seconds.float().unsqueeze(-1) * self.frequencies
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class FutureCrossAttention(nn.Module):
    """Each action state attends to every valid video token in the horizon."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.time_encoding = TimeEncoding(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        actions,
        video,
        action_times,
        video_times,
        video_valid=None,
        need_weights=False,
    ):
        if actions.ndim != 4 or video.ndim != 4:
            raise ValueError("Expected actions [B,M,A,D] and video [B,V,Q,D]")
        batch, modes, steps, dim = actions.shape
        if video.shape[0] != batch or video.shape[-1] != dim:
            raise ValueError("Action and video batch/embedding dimensions must match")
        frames, queries = video.shape[1:3]
        if not frames or not queries:
            raise ValueError("At least one video frame and token are required")
        at = expand_timestamps(action_times, batch, steps, actions.device)
        padding = None
        valid = None
        if video_valid is not None:
            valid = torch.as_tensor(
                video_valid, device=actions.device, dtype=torch.bool
            )
            if valid.shape != (batch, frames) or not valid.any(dim=1).all():
                raise ValueError("Each sample needs at least one valid video frame")
            padding = (~valid).repeat_interleave(queries, dim=1)
            video = video.masked_fill(~valid[:, :, None, None], 0)
        vt = expand_timestamps(video_times, batch, frames, actions.device, valid)
        q = actions + self.time_encoding(at).to(actions.dtype)[:, None]
        k = video + self.time_encoding(vt).to(video.dtype)[:, :, None]
        output, weights = self.attention(
            q.reshape(batch, modes * steps, dim),
            k.reshape(batch, frames * queries, dim),
            video.reshape(batch, frames * queries, dim),
            key_padding_mask=padding,
            need_weights=need_weights,
        )
        return output.reshape(batch, modes, steps, dim), weights

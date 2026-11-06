"""Spatial and temporal resampling of future video features."""

import torch
from torch import nn

from .temporal import TimeEncoding, expand_timestamps


class VideoBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.spatial = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.temporal = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, queries, features, valid):
        batch, frames, count, dim = queries.shape
        flat = queries.reshape(batch * frames, count, dim)
        keys = features.flatten(0, 1)
        flat = self.spatial_norm(
            flat + self.spatial(flat, keys, keys, need_weights=False)[0]
        )
        sequence = flat.reshape(batch, frames, count, dim).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch * count, frames, dim)
        padding = (~valid).repeat_interleave(count, dim=0)
        sequence = self.temporal_norm(
            sequence
            + self.temporal(
                sequence,
                sequence,
                sequence,
                key_padding_mask=padding,
                need_weights=False,
            )[0]
        )
        sequence = self.ffn_norm(sequence + self.ffn(sequence))
        output = sequence.reshape(batch, count, frames, dim).permute(0, 2, 1, 3)
        return output.masked_fill(~valid[:, :, None, None], 0)


class VideoResampler(nn.Module):
    """Shared frame queries support any number of timestamped future frames."""

    def __init__(
        self, input_dim=32, dim=256, queries_per_frame=14, depth=6, heads=8, dropout=0.0
    ):
        super().__init__()
        if queries_per_frame < 1 or depth < 1:
            raise ValueError("Query count and resampler depth must be positive")
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.frame_queries = nn.Parameter(torch.randn(queries_per_frame, dim) * 0.02)
        self.time_encoding = TimeEncoding(dim)
        self.layers = nn.ModuleList(
            [VideoBlock(dim, heads, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, timestamps, valid=None):
        if features.ndim != 4 or features.shape[1] < 1 or features.shape[2] < 1:
            raise ValueError("Expected nonempty video features [B,V,N,C]")
        batch, frames = features.shape[:2]
        if valid is None:
            valid = torch.ones(batch, frames, device=features.device, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid, device=features.device, dtype=torch.bool)
        if valid.shape != (batch, frames) or not valid.any(dim=1).all():
            raise ValueError("Each sample needs at least one valid frame")
        times = expand_timestamps(timestamps, batch, frames, features.device, valid)
        # Padding never contributes to either spatial or temporal representations.
        features = features.masked_fill(~valid[:, :, None, None], 0)
        projected = self.input_projection(features)
        time = self.time_encoding(times).to(projected.dtype)[:, :, None]
        projected = projected + time
        queries = self.frame_queries[None, None].expand(batch, frames, -1, -1) + time
        for layer in self.layers:
            queries = layer(queries, projected, valid)
        return self.norm(queries).masked_fill(~valid[:, :, None, None], 0)

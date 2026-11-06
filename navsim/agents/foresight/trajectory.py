"""Trajectory refinement with continuous-time future-video attention."""

import math

import numpy as np
import torch
from torch import nn

from .modules.blocks import linear_relu_ln, gen_sineembed_for_position
from .modules.multimodal_loss import LossComputer
from .spatial import GridSampleCrossBEVAttention, DiffMotionPlanningRefinementModule
from .temporal import FutureCrossAttention, future_timestamps
from .video import VideoResampler


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, heads = config.tf_d_model, config.tf_num_head
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            dim,
            heads,
            in_bev_dims=dim,
            num_points=config.trajectory_sampling.num_poses,
            config=config,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.future_attention = FutureCrossAttention(dim, heads, config.tf_dropout)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout_mode = nn.Dropout(0.1)
        self.norm1, self.norm2, self.norm_mode = (nn.LayerNorm(dim) for _ in range(3))
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.tf_d_ffn), nn.ReLU(), nn.Linear(config.tf_d_ffn, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=dim,
            ego_fut_ts=config.trajectory_sampling.num_poses,
            ego_fut_mode=config.num_trajectory_modes,
        )

    def forward(
        self,
        queries,
        points,
        bev,
        agents,
        ego,
        video,
        action_times,
        video_times,
        video_valid=None,
    ):
        batch, modes, steps, dim = queries.shape
        query = self.cross_bev_attention(queries, points, bev, bev.shape[-2:])
        query = self.norm1(
            query
            + self.dropout(
                self.cross_agent_attention(
                    query,
                    agents,
                    agents,
                    need_weights=False,
                )[0]
            )
        )
        query = self.norm2(
            query
            + self.dropout1(
                self.cross_ego_attention(
                    query,
                    ego,
                    ego,
                    need_weights=False,
                )[0]
            )
        )
        query = query.reshape(batch, modes, steps, dim)
        future, _ = self.future_attention(
            query,
            video,
            action_times,
            video_times,
            video_valid,
        )
        query = self.norm_mode(query + self.dropout_mode(future))
        query = self.norm3(self.ffn(query)).reshape(batch, modes * steps, dim)
        regression, classification = self.task_decoder(query)
        regression = torch.cat(
            (regression[..., :2] + points, regression[..., 2:3].tanh() * math.pi),
            dim=-1,
        )
        return regression, classification


class TrajectoryDecoder(nn.Module):
    def __init__(self, config, plan_anchor=None):
        super().__init__()
        self.num_steps = config.trajectory_sampling.num_poses
        self.interval = config.trajectory_sampling.interval_length
        if self.interval <= 0:
            raise ValueError("Action interval must be positive")
        anchor = (
            np.load(config.plan_anchor_path) if plan_anchor is None else plan_anchor
        )
        anchor = torch.as_tensor(anchor, dtype=torch.float32)
        if tuple(anchor.shape) != (config.num_trajectory_modes, self.num_steps, 2):
            raise ValueError(
                "Trajectory anchors must have shape [modes, action_steps, 2]"
            )
        self.plan_anchor = nn.Parameter(anchor, requires_grad=False)
        dim = config.tf_d_model
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(dim, 1, 1, 64),
            nn.Linear(dim, dim),
        )
        self.video_former = VideoResampler(
            input_dim=config.video_latent_dim,
            dim=dim,
            queries_per_frame=config.video_queries_per_frame,
            depth=config.video_qformer_depth,
            heads=config.tf_num_head,
            dropout=config.tf_dropout,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.trajectory_decoder_layers)]
        )
        if not self.layers:
            raise ValueError("At least one trajectory decoder layer is required")
        self.loss_computer = LossComputer(config)

    def forward(
        self, bev, agents, ego, video, video_times, targets=None, video_valid=None
    ):
        if self.training and targets is None:
            raise ValueError("Training trajectory refinement requires targets")
        if video.ndim == 5:
            video = video.flatten(2, 3)
        video_tokens = self.video_former(video, video_times, video_valid)
        batch = bev.shape[0]
        anchors = self.plan_anchor.unsqueeze(0).expand(batch, -1, -1, -1)
        query = self.plan_anchor_encoder(
            gen_sineembed_for_position(anchors, hidden_dim=64)
        )
        action_times = future_timestamps(
            self.num_steps, 1 / self.interval, device=bev.device
        )
        points = anchors
        losses = {}
        for index, layer in enumerate(self.layers):
            regression, classification = layer(
                query,
                points,
                bev,
                agents,
                ego,
                video_tokens,
                action_times,
                video_times,
                video_valid,
            )
            if self.training:
                losses[f"trajectory_loss_{index}"] = self.loss_computer(
                    regression,
                    classification,
                    targets,
                    anchors,
                )
            points = regression[..., :2].detach()
        best_mode = classification.argmax(dim=-1)
        best = regression[torch.arange(batch, device=bev.device), best_mode]
        output = {"trajectory": best}
        if self.training:
            output.update(
                trajectory_loss=sum(losses.values()), trajectory_loss_dict=losses
            )
        return output

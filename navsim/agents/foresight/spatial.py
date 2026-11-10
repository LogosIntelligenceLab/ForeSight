"""Spatial scene sampling and trajectory refinement."""

import numpy as np
import torch
from torch import nn
from .modules.blocks import linear_relu_ln, bias_init_with_prob


class GridSampleCrossBEVAttention(nn.Module):

    def __init__(
        self,
        embed_dims,
        num_heads,
        num_levels=1,
        in_bev_dims=64,
        num_points=8,
        config=None,
    ):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims, 1)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)
        self.value_proj = nn.Sequential(
            nn.Conv2d(
                in_bev_dims,
                embed_dims,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )
        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """
        bs, num_queries, num_points, _ = traj_points.shape
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = (
            normalized_trajectory[..., 0] / self.config.lidar_max_y
        )
        normalized_trajectory[..., 1] = (
            normalized_trajectory[..., 1] / self.config.lidar_max_x
        )
        normalized_trajectory = normalized_trajectory[..., [1, 0]]
        queries = queries.reshape(bs, num_queries * num_points, -1)
        attention_weights = self.attention_weights(queries)
        attention_weights = (
            attention_weights.squeeze(-1).view(bs, num_queries, num_points).softmax(-1)
        )
        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        sampled_features = torch.nn.functional.grid_sample(
            value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        attention_weights = attention_weights.unsqueeze(1)
        out = attention_weights * sampled_features
        out = (
            out.permute(0, 2, 3, 1)
            .reshape(bs, num_queries * num_points, -1)
            .contiguous()
        )
        out = self.output_proj(out)
        return self.dropout(out) + queries


class DiffMotionPlanningRefinementModule(nn.Module):

    def __init__(
        self, embed_dims=256, ego_fut_ts=8, ego_fut_mode=20, if_zeroinit_reg=True
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 1)
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 3),
        )
        self.if_zeroinit_reg = False
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature):
        bs, ego_fut_mode_ego_fut_point, _ = traj_feature.shape
        traj_feature = traj_feature.reshape(bs, self.ego_fut_mode, self.ego_fut_ts, -1)
        plan_reg = self.plan_reg_branch(traj_feature)
        traj_feature = traj_feature.max(dim=2)[0]
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        return (plan_reg, plan_cls)

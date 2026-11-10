"""Current-scene perception and timestamped future-conditioned planning."""

import torch
import torch.nn.functional as F
from torch import nn

from .agent_head import AgentHead
from .modules.blocks import linear_relu_ln
from .trajectory import TrajectoryDecoder
from .world_model import EponaRollout


class ForeSightModel(nn.Module):
    def __init__(self, config, backbone=None, world_model=None, plan_anchor=None):
        super().__init__()
        self._config = config
        if backbone is None:
            from .backbone import TransfuserBackbone

            backbone = TransfuserBackbone(config)
        self._backbone = backbone
        self.world_model = (
            EponaRollout.from_config(config) if world_model is None else world_model
        )
        dim = config.tf_d_model
        self._query_splits = [1, config.num_bounding_boxes]
        self._keyval_embedding = nn.Embedding(8 * 8 + 1, dim)
        self._query_embedding = nn.Embedding(sum(self._query_splits), dim)
        self._bev_downscale = nn.Conv2d(512, dim, kernel_size=1)
        self._status_encoding = nn.Linear(8, dim)
        channels = config.bev_features_channels
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, config.num_bev_classes, kernel_size=1),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(config.num_bounding_boxes, config.tf_d_ffn, dim)
        self.bev_proj = nn.Sequential(*linear_relu_ln(dim, 1, 1, dim + channels))
        self._trajectory_head = TrajectoryDecoder(config, plan_anchor=plan_anchor)

    def forward(self, features, targets=None):
        future, video_times = self.world_model(features["camera_feature_for_gen"])
        batch = features["status_feature"].shape[0]
        bev_upscale, bev, _, _ = self._backbone(
            features["camera_feature"],
            features["lidar_feature"],
        )
        spatial = bev.shape[-2:]
        context = self._bev_downscale(bev).flatten(2).transpose(1, 2)
        status = self._status_encoding(features["status_feature"])
        keyval = torch.cat((context, status[:, None]), dim=1)
        if keyval.shape[1] != self._keyval_embedding.num_embeddings:
            raise ValueError("Current-scene backbone must produce an 8x8 feature grid")
        keyval = keyval + self._keyval_embedding.weight[None]
        scene = keyval[:, :-1].transpose(1, 2).reshape(batch, -1, *spatial)
        scene = F.interpolate(
            scene, size=bev_upscale.shape[-2:], mode="bilinear", align_corners=False
        )
        scene = torch.cat((scene, bev_upscale), dim=1)
        scene = self.bev_proj(scene.flatten(2).transpose(1, 2)).transpose(1, 2)
        scene = scene.reshape(batch, -1, *bev_upscale.shape[-2:])
        queries = self._query_embedding.weight[None].expand(batch, -1, -1)
        queries = self._tf_decoder(queries, keyval)
        ego, agents = queries.split(self._query_splits, dim=1)
        output = {"bev_semantic_map": self._bev_semantic_head(bev_upscale)}
        output.update(self._agent_head(agents))
        output.update(
            self._trajectory_head(scene, agents, ego, future, video_times, targets)
        )
        return output

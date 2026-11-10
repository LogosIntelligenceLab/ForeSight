import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class ForeSightConfig:
    """Global TransFuser config."""

    trajectory_sampling: TrajectorySampling = field(
        default_factory=lambda: TrajectorySampling(time_horizon=4, interval_length=0.5)
    )

    video_fps: float = 5.0
    video_latent_dim: int = 32
    video_queries_per_frame: int = 14
    video_qformer_depth: int = 6
    trajectory_decoder_layers: int = 2
    num_trajectory_modes: int = 20
    denoising_steps: int = 100
    scheduler_epochs: int = 100
    warmup_epochs: int = 3
    epona_config_path: str = str(
        Path(__file__).parent / "epona/configs/dit_config_dcae_nuplan.py"
    )
    epona_checkpoint: str = os.environ.get(
        "FORESIGHT_EPONA_CKPT", "ckpts/epona_ckpts/epona_nuplan.pkl"
    )
    vae_checkpoint: str = os.environ.get(
        "FORESIGHT_VAE_CKPT", "ckpts/epona_ckpts/dcae_td_20000.pkl"
    )

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    bkb_path: str = os.environ.get(
        "FORESIGHT_BACKBONE_CKPT", "ckpts/resnet34/pytorch_model.bin"
    )
    plan_anchor_path: str = os.environ.get(
        "FORESIGHT_PLAN_ANCHOR", "ckpts/kmeans_navsim_traj_20.npy"
    )

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = (
        0.0  # for Attention in self._tf_decoder and self._trajectory_head
    )

    num_bounding_boxes: int = 30

    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False

    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: (
            "linestring",
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        ),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    weight_decay: float = 1e-4
    cfg_lr_mult = 0.5

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [
            self.lidar_min_x,
            self.lidar_max_x,
            self.lidar_min_y,
            self.lidar_max_y,
        ]
        return max([abs(value) for value in values])

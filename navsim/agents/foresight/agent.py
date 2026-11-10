"""NAVSIM agent interface for ForeSight."""

import logging
import math
import os

import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from .features import ForeSightFeatureBuilder, ForeSightTargetBuilder
from .loss import transfuser_loss
from .model import ForeSightModel

logger = logging.getLogger(__name__)


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    return {key.removeprefix("agent."): value for key, value in state.items()}


class ForeSightAgent(AbstractAgent):
    def __init__(
        self,
        config,
        lr=1e-4,
        checkpoint_path=None,
        pretrained_path=None,
        build_model=True,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = ForeSightModel(config) if build_model else None
        pretrained = pretrained_path or os.environ.get("FORESIGHT_PLANNER_PRETRAIN")
        if build_model and checkpoint_path is None and pretrained:
            state = checkpoint_state(pretrained)
            own = self.state_dict()
            compatible = {
                key: value
                for key, value in state.items()
                if key in own and own[key].shape == value.shape
            }
            if not compatible:
                raise ValueError(
                    "Pretrained checkpoint contains no compatible planner tensors"
                )
            self.load_state_dict(compatible, strict=False)
            logger.info("Initialized %d compatible planner tensors", len(compatible))

    def name(self):
        return "ForeSight"

    def initialize(self):
        if self._transfuser_model is None:
            raise RuntimeError("Feature-only agents cannot run evaluation")
        if not self._checkpoint_path:
            raise ValueError("Evaluation requires agent.checkpoint_path")
        self.load_state_dict(checkpoint_state(self._checkpoint_path), strict=True)
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.eval()

    def get_sensor_config(self):
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self):
        return [ForeSightTargetBuilder(config=self._config)]

    def get_feature_builders(self):
        return [ForeSightFeatureBuilder(config=self._config)]

    def forward(self, features, targets=None):
        if self._transfuser_model is None:
            raise RuntimeError("Feature-only agents cannot run model inference")
        return self._transfuser_model(features, targets)

    def compute_loss(self, features, targets, predictions):
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self):
        if self._transfuser_model is None:
            raise RuntimeError("Feature-only agents cannot run training")
        cfg = self._config
        groups = {}
        for name, parameter in self._transfuser_model.named_parameters():
            if parameter.requires_grad:
                scale = cfg.cfg_lr_mult if "image_encoder" in name else 1.0
                groups.setdefault(scale, []).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": values, "lr": self._lr * scale}
                for scale, values in groups.items()
            ],
            lr=self._lr,
            weight_decay=cfg.weight_decay,
        )
        epochs, warmup = cfg.scheduler_epochs, cfg.warmup_epochs
        if not 0 <= warmup < epochs:
            raise ValueError("Warmup epochs must be smaller than scheduler epochs")

        def schedule(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = min(1.0, (epoch - warmup) / (epochs - warmup))
            return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
        }

    def get_training_callbacks(self):
        return []

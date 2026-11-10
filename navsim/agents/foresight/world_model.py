"""Frozen native-rate Epona rollout from the current front-camera observation."""

import math

import torch
from torch import nn

from .temporal import future_timestamps


class EponaRollout(nn.Module):
    def __init__(self, backend, tokenizer, frequency=5.0, horizon=4.0):
        super().__init__()
        if not math.isfinite(frequency) or frequency <= 0 or horizon <= 0:
            raise ValueError("Video frequency and horizon must be positive")
        frames = frequency * horizon
        if not math.isfinite(frames) or not math.isclose(
            frames, round(frames), abs_tol=1e-6
        ):
            raise ValueError("Video horizon must contain a whole number of frames")
        if round(frames) < 1:
            raise ValueError("Video horizon must contain at least one future frame")
        if backend.condition_frames != 1:
            raise ValueError("Current-observation rollout requires condition_frames=1")
        self.backend = backend.requires_grad_(False)
        self.tokenizer = tokenizer.requires_grad_(False)
        self.frequency = frequency
        self.num_frames = round(frames)
        self.train(False)

    @classmethod
    def from_config(cls, config):
        # Heavy model dependencies are imported only when loading actual assets.
        from .epona.model import TrainTransformersDiT
        from .epona.modules.tokenizer import VAETokenizer
        import runpy
        from types import SimpleNamespace

        args = SimpleNamespace(
            **{
                key: value
                for key, value in runpy.run_path(
                    config.epona_config_path,
                ).items()
                if not key.startswith("__")
            }
        )
        args.resume_path = config.epona_checkpoint
        args.vae_ckpt = config.vae_checkpoint
        args.batch_size = 1
        args.condition_frames = 1
        args.num_sampling_steps = config.denoising_steps
        backend = TrainTransformersDiT(
            args, load_path=args.resume_path, condition_frames=1
        )
        return cls(
            backend,
            VAETokenizer(args, 0),
            config.video_fps,
            config.trajectory_sampling.time_horizon,
        )

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected current RGB image [B,3,H,W]")
        self.train(False)
        batch = image.shape[0]
        latents = self.tokenizer.encode_to_z((image * 2 - 1).unsqueeze(1))
        pose = image.new_zeros(batch, 2, 2)
        yaw = image.new_zeros(batch, 2, 1)
        future = []
        for _ in range(self.num_frames):
            trajectory, frame = self.backend.step_eval(
                latents, pose, yaw, self_pred_traj=True
            )
            if frame.ndim != 4 or frame.shape[0] != batch:
                raise ValueError("Epona must return frame latents [B,H,W,C]")
            if trajectory is None or trajectory.ndim != 3 or trajectory.shape[-1] != 3:
                raise ValueError("Epona must predict its own relative trajectory")
            future.append(frame)
            predicted_pose = trajectory[:, :1, :2]
            predicted_yaw = trajectory[:, :1, 2:3]
            pose = torch.cat((predicted_pose, predicted_pose), dim=1)
            yaw = torch.cat((predicted_yaw, predicted_yaw), dim=1)
            latents = frame.flatten(1, 2).unsqueeze(1)
        timestamps = future_timestamps(
            self.num_frames, self.frequency, device=image.device
        )
        return torch.stack(future, dim=1), timestamps

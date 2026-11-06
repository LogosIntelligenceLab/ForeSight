"""CPU checks of temporal fusion, native-rate rollout and planner integration."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from navsim.agents.foresight.model import ForeSightModel
from navsim.agents.foresight.temporal import (
    FutureCrossAttention,
    TimeEncoding,
    future_timestamps,
)
from navsim.agents.foresight.trajectory import TrajectoryDecoder
from navsim.agents.foresight.video import VideoResampler
from navsim.agents.foresight.world_model import EponaRollout


def config():
    return SimpleNamespace(
        tf_d_model=32,
        tf_num_head=4,
        tf_d_ffn=64,
        tf_dropout=0.0,
        tf_num_layers=1,
        trajectory_sampling=SimpleNamespace(
            num_poses=8, interval_length=0.5, time_horizon=4.0
        ),
        num_trajectory_modes=3,
        video_latent_dim=32,
        video_queries_per_frame=2,
        video_qformer_depth=1,
        trajectory_decoder_layers=2,
        num_bounding_boxes=3,
        trajectory_cls_weight=10.0,
        trajectory_reg_weight=8.0,
        lidar_max_x=32.0,
        lidar_max_y=32.0,
        bev_features_channels=64,
        num_bev_classes=7,
        lidar_resolution_height=8,
        lidar_resolution_width=8,
        trajectory_weight=12.0,
        agent_class_weight=10.0,
        agent_box_weight=1.0,
        bev_semantic_weight=14.0,
        diff_loss_weight=20.0,
        latent=False,
    )


class SmallTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def encode_to_z(self, image):
        return (
            image.mean(dim=(2, 3, 4), keepdim=False)[:, :, None, None].expand(
                -1, -1, 4, 32
            )
            * self.scale
        )


class SmallEpona(nn.Module):
    condition_frames = 1

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = 0

    def step_eval(self, latents, pose, yaw, self_pred_traj):
        self.calls += 1
        if not self_pred_traj:
            raise AssertionError("World-model rollout must predict its own motion")
        frame = latents[:, 0].reshape(-1, 2, 2, 32) + self.scale * 0.01
        trajectory = latents.new_zeros(latents.shape[0], 1, 3)
        trajectory[..., 0] = 0.1
        return trajectory, frame


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Conv2d(3, 64, 1)
        self.context = nn.Conv2d(64, 512, 1)

    def forward(self, camera, lidar):
        up = torch.nn.functional.adaptive_avg_pool2d(self.image(camera), (8, 8))
        return up, self.context(up), None, None


class TemporalPlanningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_physical_timestamps_and_encoding(self):
        video = future_timestamps(20, 5)
        action = future_timestamps(8, 2)
        self.assertEqual(video[-1].item(), 4)
        self.assertEqual(action[-1].item(), 4)
        self.assertAlmostEqual(video[0].item(), 0.2)
        encoding = TimeEncoding(32)
        torch.testing.assert_close(encoding(video[4]), encoding(action[1]))
        self.assertFalse(torch.allclose(encoding(video[0]), encoding(action[0])))

    def test_global_attention_accepts_twenty_video_and_eight_action_steps(self):
        layer = FutureCrossAttention(32, 4).eval()
        actions = torch.randn(2, 3, 8, 32, requires_grad=True)
        video = torch.randn(2, 20, 14, 32, requires_grad=True)
        output, weights = layer(
            actions,
            video,
            future_timestamps(8, 2),
            future_timestamps(20, 5),
            need_weights=True,
        )
        self.assertEqual(tuple(output.shape), (2, 3, 8, 32))
        self.assertEqual(tuple(weights.shape), (2, 24, 280))
        self.assertTrue((weights > 0).all())
        output.square().mean().backward()
        self.assertGreater(video.grad.abs().sum().item(), 0)

    def test_timestamp_values_affect_fusion(self):
        layer = FutureCrossAttention(32, 4).eval()
        actions, video = torch.randn(1, 3, 8, 32), torch.randn(1, 20, 2, 32)
        first = layer(
            actions, video, future_timestamps(8, 2), future_timestamps(20, 5)
        )[0]
        second = layer(
            actions, video, future_timestamps(8, 2), future_timestamps(20, 10)
        )[0]
        self.assertFalse(torch.allclose(first, second))

    def test_padding_cannot_change_valid_predictions(self):
        resampler = VideoResampler(32, 32, 2, depth=1, heads=4).eval()
        layer = FutureCrossAttention(32, 4).eval()
        features = torch.randn(2, 20, 4, 32)
        valid = torch.ones(2, 20, dtype=torch.bool)
        valid[:, 10:] = False
        other = features.clone()
        other[:, 10:] = 10000
        times = future_timestamps(20, 5)
        a, b = resampler(features, times, valid), resampler(other, times, valid)
        torch.testing.assert_close(a, b)
        padded_times = times.clone()
        padded_times[10:] = float("nan")
        torch.testing.assert_close(a, resampler(other, padded_times, valid))
        actions = torch.randn(2, 3, 8, 32)
        torch.testing.assert_close(
            layer(actions, a, future_timestamps(8, 2), times, valid)[0],
            layer(actions, b, future_timestamps(8, 2), times, valid)[0],
        )

    def test_one_resampler_supports_different_frame_counts(self):
        model = VideoResampler(32, 32, 2, depth=1, heads=4).eval()
        for count in [8, 20, 25]:
            output = model(torch.randn(2, count, 4, 32), future_timestamps(count, 5))
            self.assertEqual(tuple(output.shape), (2, count, 2, 32))

    def test_native_rate_rollout_is_frozen_and_covers_full_horizon(self):
        backend, tokenizer = SmallEpona(), SmallTokenizer()
        rollout = EponaRollout(backend, tokenizer, 5, 4)
        rollout.train()
        video, times = rollout(torch.rand(2, 3, 8, 8, requires_grad=True))
        self.assertEqual(tuple(video.shape), (2, 20, 2, 2, 32))
        self.assertEqual(backend.calls, 20)
        self.assertEqual(times[-1].item(), 4)
        self.assertFalse(video.requires_grad)
        self.assertFalse(backend.training)
        self.assertTrue(all(not p.requires_grad for p in rollout.parameters()))

    def test_planner_inference_needs_no_targets_and_ignores_supplied_targets(self):
        cfg = config()
        rollout = EponaRollout(SmallEpona(), SmallTokenizer(), 5, 4)
        model = ForeSightModel(
            cfg, SmallBackbone(), rollout, torch.randn(3, 8, 2)
        ).eval()
        features = {
            "camera_feature": torch.rand(2, 3, 8, 8),
            "camera_feature_for_gen": torch.rand(2, 3, 8, 8),
            "lidar_feature": torch.rand(2, 1, 8, 8),
            "status_feature": torch.randn(2, 8),
        }
        with torch.no_grad():
            first = model(features)
            second = model(features, {"trajectory": torch.randn(2, 8, 3) * 100})
        self.assertEqual(tuple(first["trajectory"].shape), (2, 8, 3))
        torch.testing.assert_close(first["trajectory"], second["trajectory"])

    def test_training_updates_fusion_but_not_world_model(self):
        cfg = config()
        rollout = EponaRollout(SmallEpona(), SmallTokenizer(), 5, 4)
        model = ForeSightModel(
            cfg, SmallBackbone(), rollout, torch.randn(3, 8, 2)
        ).train()
        features = {
            "camera_feature": torch.rand(2, 3, 8, 8),
            "camera_feature_for_gen": torch.rand(2, 3, 8, 8),
            "lidar_feature": torch.rand(2, 1, 8, 8),
            "status_feature": torch.randn(2, 8),
        }
        from navsim.agents.foresight.loss import transfuser_loss

        targets = {
            "trajectory": torch.randn(2, 8, 3),
            "agent_states": torch.randn(2, 3, 5),
            "agent_labels": torch.ones(2, 3, dtype=torch.bool),
            "bev_semantic_map": torch.zeros(2, 4, 8, dtype=torch.long),
        }
        output = model(features, targets)
        loss = transfuser_loss(targets, output, cfg)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = model._trajectory_head.video_former.frame_queries.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in rollout.parameters()))
        self.assertFalse(rollout.training)

    def test_invalid_time_metadata_is_rejected(self):
        layer = FutureCrossAttention(32, 4)
        actions, video = torch.randn(1, 3, 8, 32), torch.randn(1, 20, 2, 32)
        with self.assertRaises(ValueError):
            layer(actions, video, future_timestamps(8, 2), torch.zeros(20))
        with self.assertRaises(ValueError):
            layer(
                actions,
                video,
                future_timestamps(8, 2),
                future_timestamps(20, 5),
                torch.zeros(1, 20, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            EponaRollout(SmallEpona(), SmallTokenizer(), 5, 0.31)


if __name__ == "__main__":
    unittest.main()

"""Exercise the actual Epona autoregressive backend at small CPU dimensions."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from navsim.agents.foresight.world_model import EponaRollout


@unittest.skipUnless(
    importlib.util.find_spec("timm"), "Epona runtime dependencies required"
)
class EponaBackendTest(unittest.TestCase):
    def test_native_backend_generates_twenty_frames_without_targets(self):
        from navsim.agents.foresight.epona.model import TrainTransformersDiT

        args = SimpleNamespace(
            vae_embed_dim=4,
            patch_size=1,
            image_size=(32, 64),
            traj_len=2,
            downsample_size=32,
            pkeep=0.7,
            pose_x_vocab_size=128,
            pose_y_vocab_size=128,
            yaw_vocab_size=512,
            pose_x_bound=50,
            pose_y_bound=10,
            yaw_bound=12,
            block_size=1,
            n_layer=[1, 1, 1],
            n_head=4,
            n_embd=32,
            n_embd_dit=128,
            n_head_dit=1,
            axes_dim_dit=[16, 56, 56],
            n_embd_dit_traj=128,
            n_head_dit_traj=1,
            n_layer_traj=[1, 1],
            axes_dim_dit_traj=[16, 56, 56],
            batch_size=1,
            lambda_yaw_pose=1.0,
            num_sampling_steps=2,
        )

        class Tokenizer(nn.Module):
            def encode_to_z(self, image):
                return image.mean(dim=(2, 3, 4))[:, :, None, None].expand(-1, -1, 2, 4)

        torch.manual_seed(1)
        backend = TrainTransformersDiT(args, condition_frames=1)
        # Exercise the original Epona checkpoint layout before the rollout.
        original_state = {
            'module.' + name + '.' + key: value
            for name in ['model', 'dit', 'traj_dit']
            for key, value in getattr(backend, name).state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'epona.pkl'
            torch.save({'model_state_dict': original_state}, checkpoint)
            restored = TrainTransformersDiT(args, load_path=checkpoint, condition_frames=1)
            for key, value in backend.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
        backend = restored
        rollout = EponaRollout(backend, Tokenizer(), frequency=5, horizon=4)
        video, times = rollout(torch.rand(1, 3, 32, 64))
        self.assertEqual(tuple(video.shape), (1, 20, 1, 2, 4))
        self.assertEqual(times[-1].item(), 4)
        self.assertTrue(torch.isfinite(video).all())
        self.assertFalse(video.requires_grad)


if __name__ == "__main__":
    unittest.main()

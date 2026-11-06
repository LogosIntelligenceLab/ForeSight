"""Integration checks for the NAVSIM configuration and current-scene encoder."""

import importlib.util
import unittest

import torch

READY = bool(importlib.util.find_spec("nuplan") and importlib.util.find_spec("timm"))


@unittest.skipUnless(READY, "NAVSIM runtime dependencies required")
class NavsimEntryTest(unittest.TestCase):
    def test_hydra_can_build_a_feature_only_agent_without_weights(self):
        from hydra import initialize_config_module, compose
        from hydra.utils import instantiate

        with initialize_config_module(
            version_base=None,
            config_module="navsim.planning.script.config.training",
        ):
            cfg = compose(
                config_name="default_training",
                overrides=[
                    "train_test_split=navtrain",
                    "experiment_name=test",
                    "agent.build_model=False",
                ],
            )
            agent = instantiate(cfg.agent)
        self.assertIsNone(agent._transfuser_model)
        self.assertEqual(agent._config.video_fps, 5)
        self.assertEqual(agent._config.trajectory_sampling.num_poses, 8)
        self.assertEqual(
            agent.get_feature_builders()[0].get_unique_name(), "transfuser_feature"
        )

    def test_current_scene_backbone_and_temporal_planner_run_together(self):
        from navsim.agents.foresight.config import ForeSightConfig
        from navsim.agents.foresight.model import ForeSightModel
        from navsim.agents.foresight.world_model import EponaRollout
        from test_temporal_planning import SmallEpona, SmallTokenizer

        cfg = ForeSightConfig(
            bkb_path=None,
            video_qformer_depth=1,
            tf_d_model=32,
            tf_d_ffn=64,
            tf_num_head=4,
            trajectory_decoder_layers=1,
        )
        rollout = EponaRollout(SmallEpona(), SmallTokenizer(), 5, 4)
        model = ForeSightModel(
            cfg, world_model=rollout, plan_anchor=torch.randn(20, 8, 2)
        ).eval()
        features = {
            "camera_feature": torch.rand(1, 3, 256, 1024),
            "camera_feature_for_gen": torch.rand(1, 3, 32, 64),
            "lidar_feature": torch.rand(1, 1, 256, 256),
            "status_feature": torch.randn(1, 8),
        }
        with torch.no_grad():
            result = model(features)
        self.assertEqual(tuple(result["trajectory"].shape), (1, 8, 3))
        self.assertTrue(torch.isfinite(result["trajectory"]).all())


if __name__ == "__main__":
    unittest.main()

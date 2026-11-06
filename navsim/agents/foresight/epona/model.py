"""Epona inference backend. Upstream: Kevin-thu/Epona (MIT)."""

import torch
import torch.nn as nn
from einops import rearrange
from navsim.agents.foresight.epona.stt import SpatialTemporalTransformer
from navsim.agents.foresight.epona.flux_dit import FluxParams, FluxDiT
from navsim.agents.foresight.epona.traj_dit import TrajDiT, TrajParams
from navsim.agents.foresight.epona.modules.tokenizer import (
    poses_to_indices,
    yaws_to_indices,
)
from navsim.agents.foresight.epona.modules.sampling import prepare_ids, get_schedule


class TrainTransformersDiT(nn.Module):

    def __init__(self, args, local_rank=-1, load_path=None, condition_frames=3):
        super().__init__()
        self.local_rank = local_rank
        self.args = args
        self.condition_frames = condition_frames
        self.vae_emb_dim = self.args.vae_embed_dim * self.args.patch_size**2
        self.image_size = self.args.image_size
        self.traj_len = self.args.traj_len
        self.h, self.w = (
            self.image_size[0] // (self.args.downsample_size * self.args.patch_size),
            self.image_size[1] // (self.args.downsample_size * self.args.patch_size),
        )
        self.pkeep = args.pkeep
        self.img_token_size = self.h * self.w
        self.pose_x_vocab_size = self.args.pose_x_vocab_size
        self.pose_y_vocab_size = self.args.pose_y_vocab_size
        self.yaw_vocab_size = self.args.yaw_vocab_size
        self.pose_x_bound = self.args.pose_x_bound
        self.pose_y_bound = self.args.pose_y_bound
        self.yaw_bound = self.args.yaw_bound
        self.pose_token_size = 2 * self.args.block_size
        self.yaw_token_size = 1 * self.args.block_size
        self.traj_token_size = self.pose_token_size + self.yaw_token_size
        self.total_token_size = (
            self.img_token_size + self.pose_token_size + self.yaw_token_size
        )
        self.token_size_dict = {
            "img_tokens_size": self.img_token_size,
            "pose_tokens_size": self.pose_token_size,
            "yaw_token_size": self.yaw_token_size,
            "total_tokens_size": self.total_token_size,
        }
        self.model = SpatialTemporalTransformer(
            block_size=condition_frames * self.total_token_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            pose_x_vocab_size=self.pose_x_vocab_size,
            pose_y_vocab_size=self.pose_y_vocab_size,
            yaw_vocab_size=self.yaw_vocab_size,
            latent_size=(self.h, self.w),
            local_rank=local_rank,
            condition_frames=self.condition_frames,
            token_size_dict=self.token_size_dict,
            vae_emb_dim=self.vae_emb_dim,
            temporal_block=self.args.block_size,
        )
        self.dit = FluxDiT(
            FluxParams(
                in_channels=self.vae_emb_dim,
                out_channels=self.vae_emb_dim,
                vec_in_dim=args.n_embd * (self.total_token_size - self.img_token_size),
                context_in_dim=args.n_embd,
                hidden_size=args.n_embd_dit,
                mlp_ratio=4.0,
                num_heads=args.n_head_dit,
                depth=args.n_layer[1],
                depth_single_blocks=args.n_layer[2],
                axes_dim=args.axes_dim_dit,
                theta=10000,
                qkv_bias=True,
                guidance_embed=False,
            )
        )
        self.traj_dit = TrajDiT(
            TrajParams(
                in_channels=self.traj_token_size,
                out_channels=self.traj_token_size,
                context_in_dim=args.n_embd,
                hidden_size=args.n_embd_dit_traj,
                mlp_ratio=4.0,
                num_heads=args.n_head_dit_traj,
                depth=args.n_layer_traj[0],
                depth_single_blocks=args.n_layer_traj[1],
                axes_dim=args.axes_dim_dit_traj,
                theta=10000,
                qkv_bias=True,
                guidance_embed=False,
            )
        )
        bs = args.batch_size * condition_frames
        self.img_ids, self.cond_ids, self.traj_ids = prepare_ids(
            bs, self.h, self.w, self.total_token_size, self.traj_len
        )
        self.lambda_yaw_pose = self.args.lambda_yaw_pose
        if load_path is not None:
            state_dict = torch.load(load_path, map_location="cpu")["model_state_dict"]
            model_state_dict = self.model.state_dict()
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict["module.model." + k]
            self.model.load_state_dict(model_state_dict)
            traj_dit_state_dict = self.traj_dit.state_dict()
            if any((k.startswith("module.traj_dit.") for k in state_dict.keys())):
                for k in traj_dit_state_dict.keys():
                    traj_dit_state_dict[k] = state_dict["module.traj_dit." + k]
                self.traj_dit.load_state_dict(traj_dit_state_dict)
            dit_state_dict = self.dit.state_dict()
            for k in dit_state_dict.keys():
                dit_state_dict[k] = state_dict["module.dit." + k]
            self.dit.load_state_dict(dit_state_dict)
            print(f"Successfully load model from {load_path}")

    def denormalize_traj(self, traj_targets):
        traj_targets[..., 0:1] = (traj_targets[..., 0:1] + 1) * self.pose_x_bound / 2
        traj_targets[..., 1:2] *= self.pose_y_bound
        traj_targets[..., 2:3] *= self.yaw_bound
        return traj_targets

    @torch.no_grad()
    def step_eval(
        self,
        latents,
        rel_pose,
        rel_yaw,
        sample_last=True,
        self_pred_traj=True,
        traj_only=False,
    ):
        self.model.eval()
        pose_total = poses_to_indices(
            rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size
        )
        yaw_total = yaws_to_indices(rel_yaw, self.yaw_vocab_size)
        stt_features, pose_emb = self.model.evaluate(
            latents, pose_total, yaw_total, sample_last=sample_last
        )
        bsz = stt_features.shape[0]
        img_ids, cond_ids, traj_ids = prepare_ids(
            bsz, self.h, self.w, self.total_token_size, self.traj_len
        )
        img_ids = img_ids.to(stt_features.device)
        cond_ids = cond_ids.to(stt_features.device)
        traj_ids = traj_ids.to(stt_features.device)
        predict_traj = None
        if self_pred_traj:
            self.traj_dit.eval()
            noise_traj = torch.randn(bsz, self.traj_len, self.traj_token_size).to(
                stt_features
            )
            timesteps_traj = get_schedule(
                int(self.args.num_sampling_steps), self.traj_len
            )
            predict_traj = self.traj_dit.sample(
                noise_traj, traj_ids, stt_features, cond_ids, timesteps_traj
            )
            predict_traj = self.denormalize_traj(predict_traj)
        if traj_only:
            predict_latents = None
        else:
            if self_pred_traj:
                predict_pose, predict_yaw = (
                    predict_traj[:, 0:1, 0:2],
                    predict_traj[:, 0:1, 2:3],
                )
                predict_pose = poses_to_indices(
                    predict_pose, self.pose_x_vocab_size, self.pose_y_vocab_size
                )
                predict_yaw = yaws_to_indices(predict_yaw, self.yaw_vocab_size)
                pose_emb = self.model.get_pose_emb(predict_pose, predict_yaw)
            self.dit.eval()
            noise = torch.randn(bsz, self.img_token_size, self.vae_emb_dim).to(
                stt_features
            )
            timesteps = get_schedule(
                int(self.args.num_sampling_steps), self.img_token_size
            )
            predict_latents = self.dit.sample(
                noise, img_ids, stt_features, cond_ids, pose_emb, timesteps
            )
            predict_latents = rearrange(
                predict_latents, "b (h w) c -> b h w c", h=self.h, w=self.w
            )
        return (predict_traj, predict_latents)

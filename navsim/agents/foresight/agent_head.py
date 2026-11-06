"""Agent detection auxiliary head."""

from typing import Dict
import numpy as np
import torch
from torch import nn
from .geometry import BoundingBox2DIndex


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(self, num_agents: int, d_ffn: int, d_model: int):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )
        self._mlp_label = nn.Sequential(nn.Linear(self._d_model, 1))

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )
        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)
        return {"agent_states": agent_states, "agent_labels": agent_labels}

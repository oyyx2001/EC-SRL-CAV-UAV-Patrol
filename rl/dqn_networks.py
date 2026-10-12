from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


MASKED_Q_VALUE = -1e9


def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Set invalid actions to a large negative value before argmax/max."""
    mask = action_mask.to(device=q_values.device, dtype=torch.bool)
    return q_values.masked_fill(~mask, MASKED_Q_VALUE)


class CandidateAwareQNetwork(nn.Module):
    """Q network that scores no-launch and candidate-sortie actions separately."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        candidate_dim: int = 6,
        global_hidden_dim: int = 256,
        candidate_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.candidate_dim = int(candidate_dim)
        self.num_candidates = self.action_dim - 1
        self.global_dim = self.obs_dim - self.num_candidates * self.candidate_dim
        if self.action_dim < 2:
            raise ValueError("candidate-aware DQN requires at least no-launch plus one candidate slot")
        if self.global_dim <= 0:
            raise ValueError(
                f"invalid observation split: obs_dim={obs_dim}, action_dim={action_dim}, "
                f"candidate_dim={candidate_dim}"
            )

        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, global_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(global_hidden_dim),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim, candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.ReLU(),
        )
        self.candidate_q_head = nn.Linear(global_hidden_dim + candidate_hidden_dim, 1)
        self.no_launch_q_head = nn.Linear(global_hidden_dim, 1)

    def split_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        global_features = obs[:, : self.global_dim]
        candidate_features = obs[:, self.global_dim :].reshape(
            obs.shape[0], self.num_candidates, self.candidate_dim
        )
        return global_features, candidate_features

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        del action_mask
        global_features, candidate_features = self.split_observation(obs)
        global_embedding = self.global_encoder(global_features)
        candidate_embedding = self.candidate_encoder(candidate_features)
        expanded_global = global_embedding.unsqueeze(1).expand(-1, self.num_candidates, -1)
        candidate_q = self.candidate_q_head(
            torch.cat([expanded_global, candidate_embedding], dim=-1)
        ).squeeze(-1)
        no_launch_q = self.no_launch_q_head(global_embedding)
        return torch.cat([no_launch_q, candidate_q], dim=1)


class CandidateAwareDuelingQNetwork(CandidateAwareQNetwork):
    """Dueling candidate-aware Q network with mask-aware advantage centering."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        candidate_dim: int = 6,
        global_hidden_dim: int = 256,
        candidate_hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            candidate_dim=candidate_dim,
            global_hidden_dim=global_hidden_dim,
            candidate_hidden_dim=candidate_hidden_dim,
        )
        self.value_head = nn.Linear(global_hidden_dim, 1)
        self.candidate_advantage_head = nn.Linear(global_hidden_dim + candidate_hidden_dim, 1)
        self.no_launch_advantage_head = nn.Linear(global_hidden_dim, 1)
        del self.candidate_q_head
        del self.no_launch_q_head

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        global_features, candidate_features = self.split_observation(obs)
        global_embedding = self.global_encoder(global_features)
        candidate_embedding = self.candidate_encoder(candidate_features)
        expanded_global = global_embedding.unsqueeze(1).expand(-1, self.num_candidates, -1)

        value = self.value_head(global_embedding)
        candidate_advantage = self.candidate_advantage_head(
            torch.cat([expanded_global, candidate_embedding], dim=-1)
        ).squeeze(-1)
        no_launch_advantage = self.no_launch_advantage_head(global_embedding)
        advantage = torch.cat([no_launch_advantage, candidate_advantage], dim=1)

        if action_mask is None:
            mean_advantage = advantage.mean(dim=1, keepdim=True)
        else:
            mask = action_mask.to(device=advantage.device, dtype=advantage.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_advantage = (advantage * mask).sum(dim=1, keepdim=True) / denom
        return value + advantage - mean_advantage


class NoisyLinear(nn.Module):
    """Factorized noisy linear layer used by Rainbow-style DQN exploration."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = float(sigma_init)
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features**0.5)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features**0.5)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class CandidateAwareRainbowQNetwork(CandidateAwareDuelingQNetwork):
    """Rainbow-style candidate-aware network with dueling heads and NoisyLinear exploration."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        candidate_dim: int = 6,
        global_hidden_dim: int = 256,
        candidate_hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            candidate_dim=candidate_dim,
            global_hidden_dim=global_hidden_dim,
            candidate_hidden_dim=candidate_hidden_dim,
        )
        self.value_head = NoisyLinear(global_hidden_dim, 1)
        self.candidate_advantage_head = NoisyLinear(global_hidden_dim + candidate_hidden_dim, 1)
        self.no_launch_advantage_head = NoisyLinear(global_hidden_dim, 1)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()




def reset_noisy_layers(module: nn.Module) -> None:
    reset = getattr(module, "reset_noise", None)
    if callable(reset):
        reset()


def make_q_network(
    algo: str,
    obs_dim: int,
    action_dim: int,
    candidate_dim: int = 6,
    architecture: str = "candidate_aware",
) -> nn.Module:
    """Construct the released CS-RDQN; inherited encoder/dueling code is unchanged."""
    if str(architecture).lower() != "candidate_aware":
        raise ValueError("The release includes only the main candidate-aware CS-RDQN")
    if algo != "rainbow_dqn":
        raise ValueError("The release includes only rainbow_dqn (CS-RDQN)")
    return CandidateAwareRainbowQNetwork(obs_dim, action_dim, candidate_dim)

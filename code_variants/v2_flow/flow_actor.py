"""
Flow-matching actor for SAC.

Replaces the squashed-Gaussian SAC actor with a one-step flow-matching policy
inspired by FlowRL (NeurIPS 2025) and the MeanFlow line of work.

Action sampling: a^0 ~ N(0, sigma^2 I); a = tanh(a^0 + v_theta(features, a^0, t=0))
i.e. one Euler step of the probability flow ODE from t=0 to t=1.

Behaviour-cloning / W2 alignment loss is computed externally in the SAC train loop.
"""
from typing import Optional, Tuple, List, Type, Dict, Any

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import SACPolicy


class FlowActor(BasePolicy):
    """One-step flow-matching actor."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch=None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        noise_std: float = 1.0,
        time_dim: int = 16,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.use_sde = False
        self.features_dim = features_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init

        action_dim = get_action_dim(self.action_space)
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        v_input_dim = features_dim + action_dim + time_dim
        self.velocity_net = nn.Sequential(*create_mlp(v_input_dim, action_dim, net_arch, activation_fn))

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=False,
                log_std_init=self.log_std_init,
                features_extractor=self.features_extractor,
                noise_std=self.noise_std,
                time_dim=self.time_dim,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        return th.tensor(0.0)

    def reset_noise(self, batch_size: int = 1) -> None:
        return

    def _velocity(self, features_flat: th.Tensor, a_t: th.Tensor, t: th.Tensor) -> th.Tensor:
        t_emb = self.time_mlp(t)
        x = th.cat([features_flat, a_t, t_emb], dim=-1)
        return self.velocity_net(x)

    def sample_action(self, obs: th.Tensor, deterministic: bool = False, return_aux: bool = False):
        features = self.extract_features(obs)
        B = features.shape[0]
        if deterministic:
            a0 = th.zeros(B, self.action_dim, device=features.device)
        else:
            a0 = th.randn(B, self.action_dim, device=features.device) * self.noise_std
        t = th.zeros(B, 1, device=features.device)
        v = self._velocity(features, a0, t)
        a1 = a0 + v
        action = th.tanh(a1)
        if return_aux:
            return action, a0, v
        return action

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.sample_action(obs, deterministic=deterministic)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.sample_action(obs, deterministic=False)
        log_prob = th.zeros(action.shape[0], 1, device=action.device)
        return action, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


class FlowSACPolicy(SACPolicy):
    """SACPolicy variant that uses FlowActor instead of the squashed-Gaussian Actor."""

    def make_actor(self, features_extractor=None) -> FlowActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return FlowActor(**actor_kwargs).to(self.device)

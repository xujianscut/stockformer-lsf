"""
MeanFlow actor: one-step generative policy using average-velocity parametrization.

Follows OFQL (ICLR'26) and MeanFlowQL (AAAI'26).

Notation (OFQL convention):
  a_t = (1 - t) * a + t * eps,  t in [0, 1]
  v(a_t, t; s) = eps - a   (instantaneous velocity along the linear path)
  u(a_t, r, t; s) = (1 / (t - r)) * integral_r^t v(a_tau, tau; s) d tau
                  (average velocity from r to t)

MeanFlow Identity (Geng 2025):
  u(a_t, r, t; s) = v(a_t, t; s) - (t - r) * du/dt(a_t, r, t; s)
  where  du/dt = v * d_{a_t} u + d_t u  (chain rule, since da_t/dt = v).

One-step inference:
  a = eps - u_theta(eps, r=0, t=1; s),   eps ~ N(0, I)

Behaviour-cloning loss (used externally in the SAC train loop):
  Sample (s, a) ~ buffer, eps ~ N(0, I), t ~ U(0, 1).
  With prob lambda  set r = t (degenerates to instantaneous regression: u_tgt = v).
  Else sample r ~ U(0, t) and use MeanFlow Identity:
    u_tgt = v - (t - r) * (v * d_{a_t} u_theta + d_t u_theta)   (stop-gradient).
  Loss = E ||u_theta(a_t, r, t; s) - sg(u_tgt)||^2.
"""
from typing import Optional, Tuple, List, Type, Dict, Any

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import SACPolicy


class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: th.Tensor) -> th.Tensor:
        # t: [B, 1]
        half = self.dim // 2
        freqs = th.exp(th.arange(half, device=t.device, dtype=t.dtype) * (-9.21 / max(half - 1, 1)))
        ang = t * freqs.unsqueeze(0)
        emb = th.cat([th.sin(ang), th.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = th.cat([emb, th.zeros(emb.shape[0], self.dim - emb.shape[-1], device=t.device, dtype=t.dtype)], dim=-1)
        return self.mlp(emb)


class MeanFlowActor(BasePolicy):
    """One-step average-velocity actor (MeanFlow parametrization)."""

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

        self.r_embed = _SinusoidalTimeEmbed(time_dim)
        self.t_embed = _SinusoidalTimeEmbed(time_dim)

        v_input_dim = features_dim + action_dim + 2 * time_dim
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

    def u_velocity(self, features_flat: th.Tensor, a_in: th.Tensor, r: th.Tensor, t: th.Tensor) -> th.Tensor:
        """Average velocity field u_theta(features, a_in, r, t)."""
        r_e = self.r_embed(r)
        t_e = self.t_embed(t)
        x = th.cat([features_flat, a_in, r_e, t_e], dim=-1)
        return self.velocity_net(x)

    def sample_action(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(obs)
        B = features.shape[0]
        if deterministic:
            eps = th.zeros(B, self.action_dim, device=features.device)
        else:
            eps = th.randn(B, self.action_dim, device=features.device) * self.noise_std
        # One-step inference: a = eps - u_theta(eps, r=0, t=1)
        r = th.zeros(B, 1, device=features.device)
        t = th.ones(B, 1, device=features.device)
        u = self.u_velocity(features, eps, r, t)
        a = eps - u
        return th.tanh(a)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.sample_action(obs, deterministic=deterministic)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.sample_action(obs, deterministic=False)
        log_prob = th.zeros(action.shape[0], 1, device=action.device)
        return action, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


class MeanFlowSACPolicy(SACPolicy):
    """SACPolicy variant that uses MeanFlowActor."""

    def make_actor(self, features_extractor=None) -> MeanFlowActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MeanFlowActor(**actor_kwargs).to(self.device)

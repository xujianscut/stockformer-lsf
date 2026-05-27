"""
SDAC-SAC: Soft Diffusion Actor-Critic integrated with StockFormer
基于MAE_SAC修改，将原来的高斯Actor替换为Diffusion Actor
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from collections import OrderedDict

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from MySAC.SAC.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

from Transformer.models.transformer import Transformer_base as Transformer
from Transformer.utils.metrics import ranking_loss
from MySAC.SAC.policy_transformer import policy_transformer_stock_atten2 as policy_transformer_attn2

# 导入SDAC组件
from sdac_components.sdac_actor import SDACActor

# 导入Q网络包装器
from MySAC.SAC.q_network_wrapper import SimpleQNetworkWrapper

import pdb


class SAC(OffPolicyAlgorithm):
    """
    SDAC-SAC: Soft Actor-Critic with Diffusion Policy
    
    将原版SAC的高斯策略Actor替换为扩散策略Actor
    保留Critic和其他SAC组件不变
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # Transformer参数
        enc_in=96,
        dec_in=96,
        c_out_construction=96,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        dropout=0.05,
        transformer_device='cuda:0',
        transformer_path=None,
        critic_alpha=1,
        actor_alpha=0,
        # SDAC特有参数
        diffusion_steps=20,#20,
        beta_schedule='cosine',
        lambda_rsm=1.0,
        num_action_samples=16,
        # 修复5: 添加学习率衰减参数
        lr_decay_steps=30000,  # 学习率衰减总步数
        lr_final_ratio=0.5,    # 最终学习率比例（1e-4 -> 5e-5）
    ):

        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        
        # 设备检测
        if transformer_device.startswith('cuda') and not th.cuda.is_available():
            print("警告: CUDA不可用，自动切换到CPU模式")
            transformer_device = 'cpu'
        
        self.transformer_device = transformer_device
        
        # 保存SDAC参数（在_setup_model之前）
        self.diffusion_steps = diffusion_steps
        self.beta_schedule = beta_schedule
        self.lambda_rsm = lambda_rsm
        self.num_action_samples = num_action_samples
        
        # 修复5: 保存学习率衰减参数
        self.lr_decay_steps = lr_decay_steps
        self.lr_final_ratio = lr_final_ratio
        self.initial_learning_rate = learning_rate
        
        # 初始化Q网络包装器为None（稍后在_create_aliases中创建）
        self.q_network_wrapper = None

        if _init_setup_model:
            self._setup_model()
        
        # StockFormer Encoder (用于状态编码)
        self.state_transformer = Transformer(
            enc_in=enc_in, dec_in=dec_in, c_out=c_out_construction,
            n_heads=n_heads, e_layers=e_layers, d_layers=d_layers,
            d_model=d_model, d_ff=d_ff, dropout=dropout
        ).to(transformer_device)
        
        if transformer_path is not None:
            map_location = transformer_device if th.cuda.is_available() and transformer_device.startswith('cuda') else 'cpu'
            state_dict = th.load(transformer_path, map_location=map_location)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            self.state_transformer.load_state_dict(new_state_dict)
            print(f"Successfully load pretrained model on {map_location}... {transformer_path}")
        else:
            print("Successfully initialize transformer model...")
        
        self.transformer_optim = th.optim.Adam(self.state_transformer.parameters(), lr=learning_rate)
        self.transformer_criteria = th.nn.MSELoss()
        
        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha

        # Critic的Transformer (保持原样)
        self.critic_transformer = policy_transformer_attn2(d_model=d_model, dropout=dropout, lr=learning_rate).to(transformer_device)
        
        # ===== SDAC组件初始化 =====
        # 处理env为None的情况（加载模型时）
        if env is not None:
            action_dim = env.action_space.shape[0]
        else:
            # 加载模型时，使用默认值（稍后会被覆盖）
            action_dim = 10  # 默认股票数量
        
        # 创建Q网络包装器（只有在critic存在时才创建）
        if hasattr(self, 'critic') and self.critic is not None:
            self.q_network_wrapper = SimpleQNetworkWrapper(self.critic)
        else:
            self.q_network_wrapper = None
        
        # SDAC Actor (内部包含所有组件)
        self.sdac_actor = SDACActor(
            state_dim=d_model,
            action_dim=action_dim,
            hidden_sizes=[256, 256, 256],
            time_embed_dim=64,
            num_timesteps=self.diffusion_steps,
            beta_schedule=self.beta_schedule,
            temperature=self.lambda_rsm,
            num_action_samples=self.num_action_samples,
            device=transformer_device
        ).to(transformer_device)
        
        # SDAC优化器
        self.sdac_optimizer = th.optim.Adam(self.sdac_actor.parameters(), lr=learning_rate)
        
        # 如果Q网络包装器还没创建，现在创建
        if self.q_network_wrapper is None and hasattr(self, 'critic') and self.critic is not None:
            self.q_network_wrapper = SimpleQNetworkWrapper(self.critic)
        
        # 用于动态调整 temporal feature 维度的投影层
        self.temporal_proj_short = None
        self.temporal_proj_long = None
        self.temporal_actual_dim = None
        
        self.in_feat = enc_in
        self.d_model = d_model
        
        print("="*60)
        print("SDAC-SAC 初始化完成")
        print(f"  Diffusion Steps: {diffusion_steps}")
        print(f"  Beta Schedule: {beta_schedule}")
        print(f"  Action Samples: {num_action_samples}")
        print(f"  Lambda RSM: {lambda_rsm}")
        print("="*60)

    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # 版本3: Alpha初始值改为2.0（介于1.0和5.0之间）
            init_value = 2.0  # 尝试适中的Alpha值
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        
        # 确保Q网络包装器已创建
        if self.q_network_wrapper is None:
            self.q_network_wrapper = SimpleQNetworkWrapper(self.critic)


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        训练循环 - 使用SDAC替换原来的Actor训练
        版本3: 保守修复 + 适中Alpha (2.0)
        """
        self.policy.set_training_mode(True)
        self.state_transformer.train()
        self.sdac_actor.train()
        
        if self.temporal_proj_short is not None:
            self.temporal_proj_short.train()
        if self.temporal_proj_long is not None:
            self.temporal_proj_long.train()
        
        # 版本3: 使用原来的学习率更新方式
        optimizers = [self.critic.optimizer, self.critic_transformer.optimizer, 
                     self.transformer_optim, self.sdac_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        transformer_losses = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # 状态编码
            state, temporal_feature_short, temporal_feature_long, holding_stocks, loss_s = self._state_transfer(replay_data.observations)
            
            # ===== Critic更新 =====
            next_state, next_temporal_feature_short, next_temporal_feature_long, next_holding_stocks, loss_ns = self._state_transfer(replay_data.next_observations)
            
            with th.no_grad():
                # 使用SDAC Actor生成下一个动作
                next_critic_features = self.critic_transformer(next_state, next_temporal_feature_short, next_temporal_feature_long, next_holding_stocks)
                next_state_flat = next_state.mean(dim=1)
                next_actions = self.sdac_actor(next_state_flat, deterministic=False)
                
                # 版本3: 不添加探索噪声
                
                # 计算next Q值
                next_q_values = th.cat(self.critic_target(
                    self.critic_transformer(next_state, next_temporal_feature_short, next_temporal_feature_long, next_holding_stocks), 
                    next_actions
                ), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # 当前Q值
            current_q_values = self.critic(
                self.critic_transformer(state, temporal_feature_short, temporal_feature_long, holding_stocks), 
                replay_data.actions
            )

            # Critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # 优化Critic
            self.critic.optimizer.zero_grad()
            self.critic_transformer.optimizer.zero_grad()
            self.transformer_optim.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_transformer.optimizer.step()
            self.transformer_optim.step()

            # ===== SDAC Actor更新 =====
            # 计算critic features
            critic_features = self.critic_transformer(state, temporal_feature_short, temporal_feature_long, holding_stocks).detach()
            
            # 设置Q网络包装器的cached features
            self.q_network_wrapper.set_features(critic_features)
            
            state_flat = state.mean(dim=1)
            
            # SDAC Actor内部计算损失（保留Q值缩放）
            sdac_loss_output = self.sdac_actor.compute_loss(
                state_batch=state_flat,
                action_batch=replay_data.actions,
                q_network1=self.q_network_wrapper
            )
            
            if isinstance(sdac_loss_output, tuple):
                sdac_loss = sdac_loss_output[0]
            else:
                sdac_loss = sdac_loss_output
            
            actor_losses.append(sdac_loss.item())

            # 优化SDAC Actor
            self.sdac_optimizer.zero_grad()
            sdac_loss.backward()
            self.sdac_optimizer.step()
            
            # 清除缓存
            self.q_network_wrapper.clear_cache()

            # Transformer loss
            transformerloss = (loss_s + loss_ns) / 2
            transformer_losses.append(transformerloss.item())

            # 更新target网络
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/transformer_loss", np.mean(transformer_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SDAC_SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def predict(
        self,
        test_obs: np.ndarray,
        deterministic: bool = False,
        state: np.ndarray = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测 - 使用SDAC Actor生成动作
        """
        flag = 0
        if len(test_obs.shape) == 2:
            test_obs = np.expand_dims(test_obs, axis=0)
            flag = 1
        
        self.state_transformer.eval()
        self.sdac_actor.eval()
        
        with th.no_grad():
            obs = th.FloatTensor(test_obs).to(self.transformer_device)
            obs_tensor, temporal_short, temporal_long, holding = self._state_transfer_predict(obs)
            
            # obs_tensor shape: [batch, stock_num, d_model]
            # SDAC Actor期望: [batch, d_model]
            # 需要对stock维度进行聚合
            batch_size, stock_num, d_model = obs_tensor.shape
            state_flat = obs_tensor.mean(dim=1)  # [batch, d_model] 对股票维度取平均
            
            # 使用SDAC Actor生成动作
            actions = self.sdac_actor(state_flat, deterministic=deterministic)
            
            actions_array = actions.detach().cpu().numpy()

        if flag:
            actions_array = actions_array.squeeze(0)
        
        return actions_array, None

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "critic.optimizer", "sdac_optimizer"]
        saved_pytorch_variables = []
        return state_dicts, saved_pytorch_variables
    
    def _state_transfer_predict(self, x):
        """从原始观察中提取和转换状态特征"""
        batch_enc1 = x[:, :, :self.in_feat]

        enc_out, _, output = self.state_transformer(batch_enc1, batch_enc1)
        hidden_channel = enc_out.shape[-1]

        actual_feature_dim = x.shape[2]
        remaining_dim = actual_feature_dim - self.in_feat - 1
        
        if remaining_dim < 2:
            temporal_feature_short = th.zeros(x.shape[0], x.shape[1], hidden_channel, device=x.device)
            temporal_feature_long = th.zeros(x.shape[0], x.shape[1], hidden_channel, device=x.device)
        else:
            actual_temporal_dim = remaining_dim // 2
            start_short = self.in_feat
            end_short = start_short + actual_temporal_dim
            start_long = end_short
            end_long = start_long + actual_temporal_dim
            
            temporal_feature_short = x[:, :, start_short:end_short]
            temporal_feature_long = x[:, :, start_long:end_long]
            
            if actual_temporal_dim != hidden_channel:
                if self.temporal_proj_short is None or self.temporal_actual_dim != actual_temporal_dim:
                    self.temporal_actual_dim = actual_temporal_dim
                    self.temporal_proj_short = th.nn.Linear(actual_temporal_dim, hidden_channel).to(self.transformer_device)
                    self.temporal_proj_long = th.nn.Linear(actual_temporal_dim, hidden_channel).to(self.transformer_device)
                
                temporal_feature_short = self.temporal_proj_short(temporal_feature_short)
                temporal_feature_long = self.temporal_proj_long(temporal_feature_long)
        
        holding = x[:, :, -1:]

        return enc_out, temporal_feature_short, temporal_feature_long, holding

    def _state_transfer(self, x):
        """状态转换 - 训练时使用"""
        bs, stock_num = x.shape[0], x.shape[1]

        batch_enc1 = x[:, :, :self.in_feat]
        mask = th.ones_like(batch_enc1)
        rand_indices = th.rand(bs, stock_num).argsort(dim=-1)
        mask_indices = rand_indices[:, :int(stock_num/2)]
        batch_range = th.arange(bs)[:, None]
        mask[batch_range, mask_indices, stock_num:] = 0
        enc_inp = mask * batch_enc1

        enc_out, _, output = self.state_transformer(enc_inp, enc_inp)
        hidden_channel = enc_out.shape[-1]

        pred = output[batch_range, mask_indices, stock_num:]
        true = batch_enc1[batch_range, mask_indices, stock_num:]
        loss = self.transformer_criteria(pred, true)

        actual_feature_dim = x.shape[2]
        remaining_dim = actual_feature_dim - self.in_feat - 1
        
        if remaining_dim < 2:
            temporal_feature_short = th.zeros(x.shape[0], x.shape[1], hidden_channel, device=x.device)
            temporal_feature_long = th.zeros(x.shape[0], x.shape[1], hidden_channel, device=x.device)
        else:
            actual_temporal_dim = remaining_dim // 2
            start_short = self.in_feat
            end_short = start_short + actual_temporal_dim
            start_long = end_short
            end_long = start_long + actual_temporal_dim
            
            temporal_feature_short = x[:, :, start_short:end_short]
            temporal_feature_long = x[:, :, start_long:end_long]
            
            if actual_temporal_dim != hidden_channel:
                if self.temporal_proj_short is None or self.temporal_actual_dim != actual_temporal_dim:
                    self.temporal_actual_dim = actual_temporal_dim
                    self.temporal_proj_short = th.nn.Linear(actual_temporal_dim, hidden_channel).to(self.transformer_device)
                    self.temporal_proj_long = th.nn.Linear(actual_temporal_dim, hidden_channel).to(self.transformer_device)
                
                temporal_feature_short = self.temporal_proj_short(temporal_feature_short)
                temporal_feature_long = self.temporal_proj_long(temporal_feature_long)

        holding = x[:, :, -1:]

        return enc_out, temporal_feature_short, temporal_feature_long, holding, loss

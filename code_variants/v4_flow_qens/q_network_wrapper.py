"""
Q Network Wrapper for SDAC-StockFormer Integration

这个包装器解决了SDAC和StockFormer架构不兼容的问题：
- SDAC期望Q网络直接接受(state, action)
- StockFormer的Q网络需要先通过critic_transformer处理state
"""

import torch
import torch.nn as nn
from typing import Tuple


class StockFormerQNetworkWrapper(nn.Module):
    """
    包装StockFormer的Q网络，使其兼容SDAC的调用方式
    
    SDAC调用: q_value = q_network(state, action)
    StockFormer需要: q_value = q_network(critic_transformer(state, ...), action)
    
    这个包装器自动处理这个转换
    """
    
    def __init__(
        self,
        q_network: nn.Module,
        critic_transformer: nn.Module,
        state_transformer: nn.Module,
        device: str = "cuda:0"
    ):
        """
        Args:
            q_network: StockFormer的原始Q网络
            critic_transformer: Critic的Transformer
            state_transformer: State的Transformer（用于获取temporal features）
            device: 计算设备
        """
        super().__init__()
        self.q_network = q_network
        self.critic_transformer = critic_transformer
        self.state_transformer = state_transformer
        self.device = device
        
        # 缓存temporal features以避免重复计算
        self._cached_state = None
        self._cached_features = None
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim] 或 [batch_size, stock_num, d_model]
            action: 动作 [batch_size, action_dim]
            
        Returns:
            q_values: Q值 tuple of [batch_size, 1]
        """
        batch_size = state.shape[0]
        
        # 检查state的维度
        if len(state.shape) == 2:
            # state是扁平的 [batch_size, state_dim]
            # 需要重构为 [batch_size, stock_num, d_model]
            # 这种情况下，我们假设state_dim = stock_num * d_model
            # 但实际上SDAC传入的state可能已经是聚合后的
            # 在这种情况下，我们需要使用缓存的features
            
            if self._cached_features is not None and self._cached_features.shape[0] == batch_size:
                # 使用缓存的critic features
                critic_features = self._cached_features
            else:
                # 无法重构，返回零Q值（这不应该发生）
                raise ValueError(f"Cannot process state with shape {state.shape}. Expected 3D tensor.")
        
        elif len(state.shape) == 3:
            # state是3D的 [batch_size, stock_num, d_model]
            # 这是正常情况，直接处理
            stock_num = state.shape[1]
            d_model = state.shape[2]
            
            # 创建dummy temporal features（全零）
            # 因为在loss计算时我们不需要真实的temporal features
            temporal_short = torch.zeros(batch_size, stock_num, d_model, device=self.device)
            temporal_long = torch.zeros(batch_size, stock_num, d_model, device=self.device)
            holding = torch.zeros(batch_size, stock_num, 1, device=self.device)
            
            # 通过critic_transformer处理
            critic_features = self.critic_transformer(state, temporal_short, temporal_long, holding)
            
            # 缓存features
            self._cached_features = critic_features.detach()
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")
        
        # 调用原始Q网络
        q_values = self.q_network(critic_features, action)
        
        return q_values
    
    def set_cached_features(self, features: torch.Tensor):
        """
        设置缓存的critic features
        
        这允许外部预先计算critic features，避免重复计算
        """
        self._cached_features = features.detach()
    
    def clear_cache(self):
        """清除缓存"""
        self._cached_features = None
        self._cached_state = None


class SimpleQNetworkWrapper(nn.Module):
    """
    简化版Q网络包装器
    
    直接使用预计算的critic features，不进行任何转换
    """
    
    def __init__(self, q_network: nn.Module):
        super().__init__()
        self.q_network = q_network
        self.cached_features = None
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        使用缓存的features调用Q网络
        
        Returns:
            q_value: 单个Q值张量 [batch_size, 1]，而不是tuple
        """
        if self.cached_features is None:
            raise RuntimeError("No cached features available. Call set_features() first.")
        
        # 检查batch size是否匹配
        action_batch_size = action.shape[0]
        features_batch_size = self.cached_features.shape[0]
        
        if action_batch_size != features_batch_size:
            # SDAC在计算loss时会扩展batch（反向采样）
            # 需要重复features以匹配action的batch size
            repeat_factor = action_batch_size // features_batch_size
            if action_batch_size % features_batch_size != 0:
                raise ValueError(f"Action batch size {action_batch_size} is not a multiple of features batch size {features_batch_size}")
            
            # 重复features
            expanded_features = self.cached_features.repeat_interleave(repeat_factor, dim=0)
            q_values = self.q_network(expanded_features, action)
        else:
            q_values = self.q_network(self.cached_features, action)
        
        # Q网络返回tuple of tensors，我们取最小值（Double Q的标准做法）
        if isinstance(q_values, tuple):
            q_values = torch.cat(q_values, dim=1)
            q_value, _ = torch.min(q_values, dim=1, keepdim=True)
            return q_value
        
        return q_values
    
    def set_features(self, features: torch.Tensor):
        """设置预计算的critic features"""
        self.cached_features = features
    
    def clear_cache(self):
        """清除缓存"""
        self.cached_features = None

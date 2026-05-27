"""
V5: learnable set-attention fusion of three predictive-coding branches.

Drop-in replacement for `policy_transformer_stock_atten2`.

Motivation: the original module hard-codes the cascade order
  long-horizon  -->  short-horizon  -->  relational
and a reviewer can reasonably ask whether this ordering is optimal.

V5 treats {relational, short, long} as a *set* of three source views per
stock and lets the model learn its own fusion weights via a learnable
query token attending over the set. Same input/output shapes so the rest
of the SAC code is unchanged.
"""
import itertools

import torch
from torch import nn

from Transformer.models.attn import AttentionLayer, FullAttention


class policy_transformer_set_fusion(nn.Module):
    """Learnable set fusion of {relational, short, long} per stock."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.0,
                 lr: float = 1e-4, output_attention: bool = False, device: str = "cuda:0"):
        super().__init__()
        self.d_model = d_model
        self.proj_r = nn.Linear(d_model, d_model)
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_l = nn.Linear(d_model, d_model)
        # Position / type embedding to distinguish the three source views.
        self.type_emb = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.type_emb, std=0.02)
        # Learnable query token shared across stocks (per-stock context comes from sources).
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.query_token)
        # Cross-attention with the set of sources as memory.
        self.fusion_attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
            d_model,
            n_heads,
        )
        # Light residual self-attention across stocks to keep cross-asset info.
        self.cross_stock_attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
            d_model,
            n_heads,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device

    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, holding, mask=None):
        # All inputs: [B, N, D]; holding: [B, N, 1]
        B, N, D = relational_feature.shape

        r = self.proj_r(relational_feature)
        s = self.proj_s(temporal_feature_short)
        l = self.proj_l(temporal_feature_long)

        # Set of three sources per stock: [B, N, 3, D] -> [B*N, 3, D]
        sources = torch.stack([r, s, l], dim=2).reshape(B * N, 3, D)
        q = self.query_token.expand(B * N, 1, D)

        fused, _ = self.fusion_attn(q, sources, sources, attn_mask=None)
        fused = fused.reshape(B, N, D)
        fused = self.norm1(fused + relational_feature)  # residual into relational latent

        # Cross-stock self-attention (preserves the relational signal that the
        # original module captured via two-step cross attention).
        cs_out, _ = self.cross_stock_attn(fused, fused, fused, attn_mask=mask)
        fused = self.norm2(fused + self.dropout(cs_out))

        return torch.cat([fused, holding], dim=-1)  # [B, N, D+1]

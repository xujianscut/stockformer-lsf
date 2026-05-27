"""
V5-S: V5 set fusion + per-stock sector embedding prior.

Adds a learnable embedding for the GICS-style sector each stock belongs to,
broadcast and summed into the projected source views before fusion. Sector
mapping is loaded from `../data/CSI/sector_map.csv` based on the fixed CSI
ticker order from MySAC.config.
"""
import os
import itertools

import torch
from torch import nn
import pandas as pd

from Transformer.models.attn import AttentionLayer, FullAttention

_DEFAULT_SECTOR_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "data", "CSI", "sector_map.csv"
)
_DEFAULT_N_SECTORS = 11  # 0..10


def _load_sector_ids(ticker_list, csv_path=_DEFAULT_SECTOR_CSV):
    df = pd.read_csv(csv_path).set_index("ticker")["sector_id"].astype(int)
    return [int(df.get(t, 0)) for t in ticker_list]


class policy_transformer_set_fusion_sector(nn.Module):
    """V5 set fusion + sector embedding prior."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.0,
                 lr: float = 1e-4, output_attention: bool = False, device: str = "cuda:0",
                 ticker_list=None, n_sectors: int = _DEFAULT_N_SECTORS):
        super().__init__()
        self.d_model = d_model
        self.proj_r = nn.Linear(d_model, d_model)
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_l = nn.Linear(d_model, d_model)
        self.type_emb = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.type_emb, std=0.02)

        # Sector embedding prior.
        if ticker_list is None:
            from MySAC import config
            ticker_list = config.use_ticker_dict["CSI"]
        sector_ids = _load_sector_ids(ticker_list)
        self.register_buffer("sector_ids", torch.tensor(sector_ids, dtype=torch.long))
        self.sector_emb = nn.Embedding(n_sectors, d_model)
        nn.init.normal_(self.sector_emb.weight, std=0.02)

        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.query_token)
        self.fusion_attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads,
        )
        self.cross_stock_attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device

    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, holding, mask=None):
        B, N, D = relational_feature.shape
        sec_e = self.sector_emb(self.sector_ids).unsqueeze(0)  # [1, N, D]

        r = self.proj_r(relational_feature) + self.type_emb[0] + sec_e
        s = self.proj_s(temporal_feature_short) + self.type_emb[1] + sec_e
        l = self.proj_l(temporal_feature_long) + self.type_emb[2] + sec_e

        sources = torch.stack([r, s, l], dim=2).reshape(B * N, 3, D)
        q = self.query_token.expand(B * N, 1, D)

        fused, _ = self.fusion_attn(q, sources, sources, attn_mask=None)
        fused = fused.reshape(B, N, D)
        fused = self.norm1(fused + relational_feature)

        cs_out, _ = self.cross_stock_attn(fused, fused, fused, attn_mask=mask)
        fused = self.norm2(fused + self.dropout(cs_out))

        return torch.cat([fused, holding], dim=-1)

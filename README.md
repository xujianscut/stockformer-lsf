# StockFormer-LSF: Learnable Set Fusion for Hybrid Trading Machines

Code, data, and pretrained checkpoints for our paper

> **Representation Fusion Outpaces Policy Expressivity in Price-Volume-Only Trading RL**
> Shuaibo Qiu, Jian Xu (corresponding) — *under review at IEEE ICDM 2026*

We propose a small architectural change to the [StockFormer](https://github.com/gsyyysg/StockFormer)
(IJCAI'23) hybrid trading framework: a **learnable set-fusion (LSF)** module that
replaces the hand-coded cascaded multi-head attention used to fuse the three
predictive-coding views (relational / short-horizon / long-horizon) into a
single per-stock state. The module is a drop-in replacement; nothing else in
the SAC pipeline or the Phase-1 predictive-coding encoders changes.

On three independent stock markets (CSI-300, NASDAQ-100, Nikkei 225) the LSF
variants V5 and V5a deliver consistent improvements over the StockFormer V0
baseline on portfolio return (PR), annual return (AR), Sharpe ratio (SR) and
turnover. Six algorithm-side variants (Q-ensembles, FlowRL, MeanFlow Q-learning,
and their combinations) tested on CSI-300 yield no improvement or hurt
returns — supporting the setting-specific claim in the title: within the
StockFormer pipeline on CSI-300, representation fusion is a more effective lever
than policy-class replacement.

## Repository layout

```
.
├── README.md
├── LICENSE
├── paper/paper.pdf                     the current draft
│
├── code/                               StockFormer pipeline with LSF
│   ├── train_rl.py                     Phase-2 SAC entry (CSI by default)
│   ├── eval_variant.py                 test-set evaluation with 10 metrics
│   ├── build_nasdaq.py                 yfinance NASDAQ-100 builder
│   ├── build_nikkei.py                 yfinance Nikkei 225 builder
│   ├── MySAC/                          SAC + LSF + policy_transformer_v5.py
│   ├── Transformer/
│   │   ├── pretrained/
│   │   │   ├── csi/   {mae,Short,Long}/checkpoint.pth
│   │   │   ├── nasdaq/ ...
│   │   │   └── nikkei/ ...
│   │   └── ...
│   ├── envs/                           StockFormer trading env
│   └── stable_baselines3/              vendored SB3 (matches original release)
│
├── code_variants/                      Drop-in modules for the ablation table
│   ├── v5a_notype/                     V5a (LSF without source-type embeddings)
│   ├── v5s_sector/                     V5S (+ per-stock GICS-style sector embedding)
│   ├── v5b_h8/                         V5b (8 attention heads)
│   ├── v5d_nocs/                       V5d (without cross-stock attention)
│   ├── v1_qens/                        V1 N=5 Q-ensemble + EDAC
│   ├── v2_flow/                        V2 FlowRL one-step flow actor
│   ├── v3_meanflow/                    V3 MeanFlow average-velocity actor
│   ├── v4_flow_qens/                   V4 = V1 + V2
│   └── v6_flow_fusion/                 V6 = V5 + V2
│
├── data/
│   ├── CSI/      88 CSI-300 stocks (from the official StockFormer release)
│   ├── NASDAQ/   75 NASDAQ-100 stocks (>= 98% trading days in 2011-2018)
│   └── NIKKEI/   133 Nikkei 225 stocks (>= 98% trading days in 2011-2018)
│
└── eval_out/                           Test-set metrics CSV per dataset
    ├── all_csi.csv
    ├── all_nasdaq.csv
    └── all_nikkei.csv
```

## The LSF module in one place

[`code/MySAC/SAC/policy_transformer_v5.py`](code/MySAC/SAC/policy_transformer_v5.py)
contains the entire architectural change. Given three predictive-coding
latents $s^{\mathrm{rel}}, s^{\mathrm{s}}, s^{\mathrm{l}} \in \mathbb{R}^{D}$
per stock, the module:

1. projects each through its own linear layer and adds a learnable
   source-type embedding $e_{\tau}$;
2. attends from a single learnable query token $q$ to the
   three-element set $\{r_n, u_n, v_n\}$ per stock;
3. applies a residual cross-stock self-attention to preserve inter-asset
   correlation structure;
4. concatenates the current holding $b_n$ to form the per-stock state.

V5 uses the source-type embedding, V5a drops it; both are reported as "ours".

## Quick start

```bash
# 1. clone & install
git clone https://github.com/xujianscut/stockformer-lsf.git
cd stockformer-lsf

# 2. python env (Linux/CUDA recommended)
conda create -n sflsf python=3.10 -y
conda activate sflsf
pip install torch numpy pandas scikit-learn yfinance stockstats matplotlib \
            tensorboardX shortuuid pyyaml

# 3. Pandas 2.0 hotfix for the original StockFormer preprocessor
# (replace DataFrame.append with pd.concat at lines 149 and 185)
# already applied in this repo

# 4. Phase-2 SAC training on CSI-300 (Phase-1 weights bundled under
#    code/Transformer/pretrained/csi/)
cd code
python train_rl.py

# 5. Evaluate the best checkpoint on the held-out test split
python eval_variant.py
```

To switch dataset, change `version`, `model_name`, `full_stock_dir`,
`ticker_list = config.use_ticker_dict[<...>]`, and the three Phase-1
checkpoint paths in `train_rl.py`. See the `code_v*_nasdaq_*` and
`code_v*_nikkei_*` patterns in our ICDM paper for the exact diffs.

## Results (test split, 668 / 697 / 669 trading days respectively)

Multi-seed best per method (full per-seed numbers in `eval_out/*.csv`):

| Dataset    | Method | PR$\uparrow$ | AR$\uparrow$ | SR$\uparrow$ | MDD$\downarrow$ | Calmar$\uparrow$ | Turnover$\downarrow$ |
|------------|--------|------|------|------|------|--------|--------|
| CSI-300    | V0     | 1.87 | 0.49 | 1.59 | 0.28 | 1.76 | 0.21 |
|            | V5     | **2.29** | **0.57** | 1.52 | 0.26 | 2.22 | 0.07 |
|            | V5a    | 2.23 | 0.55 | **1.82** | **0.25** | 2.19 | **0.05** |
| NASDAQ-100 | V0     | 1.11 | 0.31 | **1.21** | 0.29 | 1.08 | 0.75 |
|            | V5     | 1.18 | 0.33 | 1.19 | 0.32 | 1.01 | 0.19 |
|            | V5a    | **1.22** | **0.33** | 1.20 | **0.29** | **1.03** | **0.33** |
| Nikkei 225 | V0     | 0.29 | 0.10 | 0.63 | **0.16** | **0.50** | 0.003 (passive) |
|            | V5     | **0.33** | **0.11** | **0.67** | 0.31 | 0.37 | 0.36 |
|            | V5a    | 0.31 | 0.11 | 0.65 | 0.30 | 0.36 | 0.32 |

On Nikkei the V0 baseline collapses to a near-passive strategy with
turnover $0.003$, which mechanically yields a small MDD; V5 and V5a
maintain non-trivial turnover and deliver $+13\text{--}18\%$ higher
mean return.

## Acknowledging the StockFormer release

This work builds directly on the public StockFormer release at
<https://github.com/gsyyysg/StockFormer> (commit `9e8f1ac`). We
inherit their CSI-300 CSVs, Phase-1 transformer hyperparameters, and
SAC backbone unchanged. The contribution of this repository is the
single learnable-set-fusion module in
[`policy_transformer_v5.py`](code/MySAC/SAC/policy_transformer_v5.py)
plus the data/training scripts for the NASDAQ-100 and Nikkei 225
benchmarks.

## Cite

```bibtex
@inproceedings{qiu2026lsf,
  title={Representation Fusion Outpaces Policy Expressivity in Price--Volume-Only Trading RL},
  author={Qiu, Shuaibo and Xu, Jian},
  booktitle={Under review at IEEE ICDM},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).

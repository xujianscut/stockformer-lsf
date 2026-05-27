"""
Evaluate a StockFormer variant on the test set.

Usage:
    cd /work1/jianxu/StockFormer/<variant_code_dir>
    python eval_variant.py

Loads `trained_models/CSI/<model_name>/best_model.zip` (model_name auto-detected from
train_rl.py), runs the agent on the test environment that matches the paper's
test split, computes PR / AR / SR / MDD and writes a one-line CSV to
`results/test_metrics.csv` plus a daily-asset trajectory CSV.
"""
import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import torch as th
from sklearn.preprocessing import StandardScaler

# Local imports (variant-specific MAE_SAC etc).
sys.path.insert(0, os.getcwd())
from MySAC import config
from MySAC.preprocessors import FeatureEngineer, data_split
from MySAC.models.DRLAgent import DRLAgent, MODELS
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env


def detect_model_name(train_script: str = "train_rl.py") -> str:
    with open(train_script) as f:
        for line in f:
            m = re.match(r"^\s*model_name\s*=\s*['\"]([^'\"]+)['\"]", line)
            if m:
                return m.group(1).rstrip("/")
    raise RuntimeError("Could not detect model_name from train_rl.py")


def daily_return_from_assets(assets: np.ndarray) -> np.ndarray:
    return assets[1:] / assets[:-1] - 1.0


def max_drawdown(assets: np.ndarray) -> float:
    peak = np.maximum.accumulate(assets)
    dd = (peak - assets) / peak
    return float(dd.max())


def metrics_from_assets(assets, actions=None, n_trading_days_per_year: int = 252):
    """Compute PR, AR, SR, MDD plus Calmar / Sortino / Vol / Turnover / Hit / CVaR.

    actions: optional pandas DataFrame of daily portfolio weights (rows=days,
             cols=stocks). Used for turnover. If None, turnover is reported as NaN.
    """
    assets = np.asarray(assets, dtype=np.float64)
    n = len(assets) - 1
    rets = daily_return_from_assets(assets)

    # Core return / risk
    pr = float(assets[-1] / assets[0] - 1.0)
    ar = float((1.0 + pr) ** (n_trading_days_per_year / n) - 1.0)
    mean_r = rets.mean()
    std_r = rets.std(ddof=1) + 1e-12
    sr = float(mean_r / std_r * np.sqrt(n_trading_days_per_year))
    mdd = max_drawdown(assets)

    # Calmar = AR / MDD (return per unit drawdown)
    calmar = float(ar / mdd) if mdd > 1e-9 else float("inf")

    # Sortino = mean / downside_std (penalises only negative returns)
    downside = rets[rets < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(mean_r / (downside_std + 1e-12) * np.sqrt(n_trading_days_per_year))

    # Annualised volatility of daily returns
    vol = float(std_r * np.sqrt(n_trading_days_per_year))

    # Hit rate: fraction of strictly positive daily returns
    hit = float((rets > 0).mean())

    # CVaR at 5% level (mean of worst 5% daily returns; negative = loss)
    k = max(1, int(0.05 * len(rets)))
    cvar = float(np.sort(rets)[:k].mean())

    # Turnover: average daily L1 change of portfolio weights
    turnover = float("nan")
    if actions is not None and len(actions) > 1:
        a = np.asarray(actions, dtype=np.float64)
        # normalise rows to weights (avoid div-by-0)
        s = np.abs(a).sum(axis=1, keepdims=True) + 1e-12
        w = a / s
        turnover = float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean())

    return dict(PR=pr, AR=ar, SR=sr, MDD=mdd,
                Calmar=calmar, Sortino=sortino, Vol=vol,
                HitRate=hit, CVaR5=cvar, Turnover=turnover,
                n_days=int(n))


def main():
    model_name = detect_model_name()
    print(f"=== Evaluating variant: {model_name} ===")

    # Load data the same way train_rl.py does.
    short_prediction_model_path = 'Transformer/pretrained/csi/Short/checkpoint.pth'
    long_prediction_model_path = 'Transformer/pretrained/csi/Long/checkpoint.pth'
    ticker_list = config.use_ticker_dict['CSI']
    prediction_len = [1, 5]
    full_stock_dir = '../data/CSI/'

    df = pd.DataFrame([], columns=['date', 'open', 'close', 'high', 'low', 'volume',
                                   'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price', 'tic'])
    for ticker in ticker_list:
        temp_df = pd.read_csv(os.path.join(full_stock_dir, ticker + '.csv'),
                              usecols=['date', 'open', 'close', 'high', 'low', 'volume',
                                       'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price'])
        temp_df['date'] = pd.to_datetime(temp_df['date'].astype(str))
        temp_df['label_short_term'] = temp_df['close'].pct_change(periods=prediction_len[0]).shift(-prediction_len[0])
        temp_df['label_long_term'] = temp_df['close'].pct_change(periods=prediction_len[1]).shift(-prediction_len[1])
        temp_df['tic'] = ticker
        df = pd.concat((df, temp_df))
    df = df.sort_values(by=['date', 'tic'])

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                         use_turbulence=False, user_defined_feature=False)
    print("generate technical indicator...")
    df = fe.preprocess_data(df)

    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]
    lookback = 252
    cov_list, return_list = [], []
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        cov_list.append(return_lookback.cov().values)
    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list,
                           'return_list': return_list})
    df = df.merge(df_cov, on='date').sort_values(['date', 'tic']).reset_index(drop=True)

    scaler = StandardScaler()
    df_data = df[config.TECHNICAL_INDICATORS_LIST].replace([np.inf], config.INF).replace([-np.inf], -config.INF)
    df[config.TECHNICAL_INDICATORS_LIST] = scaler.fit_transform(df_data.values)

    test = data_split(df, '2019-01-02', '2021-12-31')
    print(f"Test split: {len(test.date.unique())} unique days, "
          f"{test.date.min()} to {test.date.max()}")

    stock_dimension = len(test.tic.unique())
    env_kwargs_test = {
        "hmax": 100,
        "initial_amount": 100000,
        "transaction_cost_pct": 0,
        "state_space": stock_dimension,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "temporal_feature_list": config.TEMPORAL_FEATURE,
        "additional_list": config.ADDITIONAL_FEATURE,
        "action_space": stock_dimension,
        "reward_scaling": 10,
        "figure_path": 'results/figures/',
        "logs_path": 'results/logs/',
        "csv_path": 'results/csv/',
        "mode": 'test',
        "time_window_start": config.time_window_start,
        "step_len": 1000,
        "temporal_len": 60,
        "hidden_channel": 128,
        "model_name": model_name,
        "short_prediction_model_path": short_prediction_model_path,
        "long_prediction_model_path": long_prediction_model_path,
    }

    test_env = Env(df=test, **env_kwargs_test)

    model_path = os.path.join('trained_models', 'CSI', model_name, 'best_model.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.zip not found at {model_path}")
    print(f"Loading model: {model_path}")

    results = DRLAgent.DRL_prediction_load_from_file(
        model_name='maesac', environment=test_env, cwd=model_path, deterministic=True
    )
    episode_total_assets = np.array(results[0], dtype=np.float64)
    assets_his = results[1]
    actions_his = results[2]

    # actions_his is a DataFrame; drop the date column to get [days, stocks].
    try:
        actions_mat = actions_his.drop(columns=['date'], errors='ignore').values
    except Exception:
        actions_mat = None
    metrics = metrics_from_assets(episode_total_assets, actions=actions_mat)
    print(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = os.path.join('results', 'test_eval')
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{**dict(variant=model_name), **metrics}]).to_csv(
        os.path.join(out_dir, 'metrics.csv'), index=False
    )
    pd.DataFrame({'day': range(len(episode_total_assets)),
                  'total_asset': episode_total_assets}).to_csv(
        os.path.join(out_dir, 'assets.csv'), index=False
    )
    print(f"Wrote {out_dir}/metrics.csv and assets.csv")


if __name__ == '__main__':
    main()

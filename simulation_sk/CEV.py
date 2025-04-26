import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------
# 0. 读入并整理原始表格
# ----------------------------------------------------------------
file_path = Path("close_2.csv")
raw = pd.read_csv(file_path, parse_dates=["Date"])

# 去掉多余 "Unnamed: n" 列
raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")],
               errors="ignore")
# 后续再做排序、去重等操作
raw = (
    raw.sort_values(["Ticker", "Date"])
        .drop_duplicates(subset=["Ticker", "Date"])
        .reset_index(drop=True)
)

# ----------------------------------------------------------------
# 1. 全局参数
# ----------------------------------------------------------------
dt                    = 1 / 252             # 日度步长
days_to_simulate      = 252                 # 未来 1 年
n_paths_per_ticker    = 10_000              # 每只资产 Monte-Carlo 条数
use_risk_neutral_mu   = False               # True → μ=r-q；False → 历史均值
div_yield_default     = 0.0                 # 分红先设定为0

# ----------------------------------------------------------------
# 2. 准备一个存放结果的 dict
# ----------------------------------------------------------------
all_simulated = {}

# ----------------------------------------------------------------
# 3. 对每只 Ticker 循环估参 + 模拟
# ----------------------------------------------------------------
tickers = raw["Ticker"].unique()
eps     = 1e-20                              # 防 log(0)

for tic in tickers:
    data    = raw.loc[raw["Ticker"] == tic].copy()
    data    = data.sort_values("Date")

    price   = data["Close"]
    spot    = price.iloc[-1]                 # S0
    rf_rate = data["rf"].dropna().iloc[-1] if data["rf"].notna().any() else 0.0
    div_y   = div_yield_default              # 也可单独为每只票指定

    # ---------- 3.1 估计 γ 与 σ：回归 log(ΔS²) ~ log(S_{t-1}) ----------
    delta_S = price.diff().dropna()
    S_prev  = price.shift(1).dropna()
    y = np.log(np.square(delta_S) + eps)
    x = np.log(S_prev.loc[y.index])
    # 线性回归 [截距, 斜率]
    Xmat = np.vstack([np.ones_like(x), x]).T
    beta0, beta1 = np.linalg.lstsq(Xmat, y.values, rcond=None)[0]
    gamma_hat  = beta1 / 2                            # γ
    sigma_hat  = np.exp(0.5 * (beta0 - np.log(dt)))   # 日度 σ

    # ---------- 3.2 估计 μ ----------
    if use_risk_neutral_mu:
        mu_annual = rf_rate - div_y
    else:
        logret    = data["log_Return"].dropna()
        mu_daily  = logret.mean() / dt + 0.5 * sigma_hat**2
        mu_annual = mu_daily * 252
    mu_daily = mu_annual / 252                        # μ

    # ---------- 3.3 构造模拟函数（单资产） ----------
    def cev_path_sim(s0, mu, sigma, gamma, n_steps=252, n_batch=1, dt=1/252):
        paths = np.zeros((n_batch, n_steps + 1))
        paths[:, 0] = s0
        for t in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_batch)
            dS = mu * paths[:, t-1] * dt + sigma * np.power(paths[:, t-1], gamma) * dW
            paths[:, t] = np.maximum(paths[:, t-1] + dS, 1e-8)
        return paths

    sim_paths = cev_path_sim(
        spot, mu_daily, sigma_hat, gamma_hat,
        n_steps=days_to_simulate,
        n_batch=n_paths_per_ticker,
        dt=dt
    )

    # 存到 dict，方便后续组合分析 / 画图
    all_simulated[tic] = {
        "gamma": gamma_hat,
        "sigma_daily": sigma_hat,
        "mu_daily": mu_daily,
        "paths": sim_paths        # ndarray (n_paths, days+1)
    }

    print(f"[{tic}] γ={gamma_hat:.3f}, σ(daily)={sigma_hat:.4f}, μ(daily)={mu_daily:.6f}")

# ----------------------------------------------------------------
# 4. 把均值路径汇总成一个 DataFrame
# ----------------------------------------------------------------
mean_paths_df = pd.DataFrame(
    {tic: all_simulated[tic]["paths"].mean(axis=0) for tic in tickers}
)
mean_paths_df.index.name = "Step"
mean_paths_df.to_csv("CEV_mean_paths_one_year.csv")
print("\n均值路径已保存：CEV_mean_paths_one_year.csv")



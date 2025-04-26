import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# 0. 读取 CSV（结构：Date, Ticker, Close, Return, log_Return, rf, …）
# ----------------------------------------------------------
file_path = Path("close_2.csv")
raw = pd.read_csv(file_path, parse_dates=["Date"])

# 去掉多余 "Unnamed: n" 列
raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")],
               errors="ignore")

# ----------------------------------------------------------
# 1. 定义单资产 GBM 模拟函数
# ----------------------------------------------------------
def simulate_gbm_paths(spot, mu, sigma, days, n_paths, dt=1/252):
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = spot
    for t in range(1, days + 1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:, t] = paths[:, t-1] * (1 + mu * dt + sigma * dW)
    return paths          # shape = (n_paths, days+1)

# ----------------------------------------------------------
# 2. 全局配置
# ----------------------------------------------------------
days       = 252               # 未来 1 年
n_paths    = 10_000            # 每只资产跑 1 万条路径
use_rn_mu  = False             # True=风险中性; False=历史均值
div_yield  = 0.0               # 分红先设定为0

# ----------------------------------------------------------
# 3. 对每个 Ticker 估参并模拟
# ----------------------------------------------------------
results = {}                   # 存模拟结果
dt      = 1/252

for tic, grp in raw.groupby("Ticker"):
    grp = grp.sort_values("Date").reset_index(drop=True)

    # --- spot ---
    spot = grp["Close"].iloc[-1]

    # --- sigma (daily) ---
    sigma_daily = grp["log_Return"].std(ddof=1)     # 样本标准差

    # --- mu (daily) ---
    if use_rn_mu:
        rf_rate = grp["rf"].dropna().iloc[-1] if grp["rf"].notna().any() else 0.0
        mu_daily = (rf_rate - div_yield) / 252
    else:
        r_bar    = grp["log_Return"].mean()
        mu_daily = r_bar/dt + 0.5 * sigma_daily**2

    # --- Monte-Carlo 模拟 ---
    paths = simulate_gbm_paths(spot, mu_daily, sigma_daily,
                               days=days, n_paths=n_paths, dt=dt)

    results[tic] = {
        "mu_daily"   : mu_daily,
        "sigma_daily": sigma_daily,
        "paths"      : paths         # ndarray (n_paths, days+1)
    }

    print(f"[{tic}] μ={mu_daily:.6f}  σ={sigma_daily:.4f}")

# ----------------------------------------------------------
# 4. 把均值路径导出成 CSV
# ----------------------------------------------------------
mean_paths = pd.DataFrame(
    {tic: res["paths"].mean(axis=0) for tic, res in results.items()}
)
mean_paths.index.name = "Step"
mean_paths.to_csv("GBM_mean_paths_one_year.csv")
print("\n所有资产的均值路径已保存：GBM_mean_paths_one_year.csv")



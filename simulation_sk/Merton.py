import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# 0 读取原表
# ----------------------------------------------------------
file_path = Path("close_2.csv")          
raw = pd.read_csv(file_path, parse_dates=["Date"])

# 去掉多余 "Unnamed: n" 列
raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")],
               errors="ignore")

# ----------------------------------------------------------
# 1 模型函数
# ----------------------------------------------------------
def simulate_merton_paths(spot, mu, sigma, lamb,
                          jump_mu, jump_sigma, days,
                          n_paths, dt=1/252):
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = spot
    for t in range(1, days + 1):
        dW   = np.random.normal(0, np.sqrt(dt), n_paths)
        J    = np.random.poisson(lamb * dt, n_paths)          # 0/1/2 … 次跳
        jumps = np.random.normal(jump_mu, jump_sigma, n_paths) * J
        paths[:, t] = paths[:, t-1] * (1 + mu * dt + sigma * dW + jumps)
    return paths

# ----------------------------------------------------------
# 2 全局设置
# ----------------------------------------------------------
days      = 252
n_paths   = 10_000
k_threshold = 3                      # 跳日判定阈值 k·σ
use_rn_mu   = False                  # True→风险中性; False→历史均值
div_yield   = 0.0

dt = 1/252

# ----------------------------------------------------------
# 3 循环每只票估参 + 模拟
# ----------------------------------------------------------
results = {}

for tic, grp in raw.groupby("Ticker"):
    grp = grp.sort_values("Date").reset_index(drop=True)
    r   = grp["log_Return"].dropna()          # 日对数收益
    spot = grp["Close"].iloc[-1]

    # ----- 初步 σ, μ (全样本) -----
    r_bar = r.mean()
    sigma_full = r.std(ddof=1)

    # ----- 标记跳日 -----
    jump_mask = (np.abs(r - r_bar) > k_threshold * sigma_full)
    jump_r    = r[jump_mask]
    non_jump_r = r[~jump_mask]

    # λ：日度跳强度
    lamb_daily = jump_mask.mean()            # (#JumpDays / TotalDays)

    # 跳均值 & 跳波动
    jump_mu    = jump_r.mean()  if len(jump_r) else 0.0
    jump_sigma = jump_r.std(ddof=1) if len(jump_r) > 1 else 0.0

    # σ：扩散波动（只看非跳日）
    sigma_daily = non_jump_r.std(ddof=1)

    # μ：选择历史 or 风险中性
    if use_rn_mu:
        rf = grp["rf"].dropna().iloc[-1] if grp["rf"].notna().any() else 0.0
        mu_daily = (rf - div_yield) / 252
    else:
        mu_daily = r_bar/dt + 0.5 * sigma_daily**2 + lamb_daily * jump_mu

    # ----- Monte-Carlo -----
    paths = simulate_merton_paths(
        spot, mu_daily, sigma_daily, lamb_daily,
        jump_mu, jump_sigma, days, n_paths, dt
    )

    results[tic] = dict(mu=mu_daily, sigma=sigma_daily,
                        lamb=lamb_daily, jump_mu=jump_mu,
                        jump_sigma=jump_sigma, paths=paths)

    print(f"[{tic}] μ={mu_daily:.6f}  σ={sigma_daily:.4f}  λ={lamb_daily:.4f}  "
          f"Jμ={jump_mu:.4f}  Jσ={jump_sigma:.4f}")

# ----------------------------------------------------------
# 4 保存均值路径
# ----------------------------------------------------------
mean_df = pd.DataFrame({tic: res["paths"].mean(axis=0) for tic, res in results.items()})
mean_df.index.name = "Step"
mean_df.to_csv("Merton_mean_paths_one_year.csv")
print("\n均值路径已保存：Merton_mean_paths_one_year.csv")



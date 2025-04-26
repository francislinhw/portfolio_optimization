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
    
# ------------------------------------------------------------
# 1 单资产 Heston Euler–Maruyama 函数
# ------------------------------------------------------------
def simulate_heston_paths(spot, mu, v0, kappa, theta, xi, rho,
                          days=252, n_paths=10_000, dt=1/252):
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = spot
    v = np.full(n_paths, v0)

    for t in range(1, days + 1):
        dW1 = np.random.normal(0, np.sqrt(dt), n_paths)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(
            0, np.sqrt(dt), n_paths
        )
        v  = np.maximum(v + kappa * (theta - v) * dt + xi * np.sqrt(v) * dW2, 1e-8)
        dS = mu * paths[:, t-1] * dt + np.sqrt(v) * paths[:, t-1] * dW1
        paths[:, t] = paths[:, t-1] + dS

    return paths        # ← 注意：同 for 循环对齐


# ------------------------------------------------------------
# 2 配置
# ------------------------------------------------------------
dt             = 1/252
days_to_sim    = 252          # 一年
n_paths        = 10_000
use_rn_mu      = False        # True→μ=r-q；False→历史均值
window_v0      = 20           # 用最近 20 天方差做 v0

# ------------------------------------------------------------
# 3 循环每只 Ticker 估参 + 模拟
# ------------------------------------------------------------
results = {}
for tic, grp in raw.groupby("Ticker"):
    grp = grp.sort_values("Date").reset_index(drop=True)

    # --- 基础序列 ---
    r      = grp["log_Return"].dropna()
    v_ts   = r ** 2                             # 即时方差序列
    v_ts_l = v_ts[:-1].values                   # v_t
    v_ts_f = v_ts[1:].values                   # v_{t+1}

    # --- μ ---
    if use_rn_mu:
        rf  = grp["rf"].dropna().iloc[-1] if grp["rf"].notna().any() else 0.0
        mu_daily = rf / 252                    # 未扣分红；如需 q 自行调整
    else:
        r_bar = r.mean()
        mu_daily = r_bar / dt + 0.5 * v_ts.mean()

    # --- v0 ---
    v0 = v_ts.tail(window_v0).mean()

    # --- kappa & theta：OLS 回归 ---
    y = (v_ts_f - v_ts_l) / dt
    X = v_ts_l
    Xmat = np.vstack([np.ones_like(X), X]).T
    beta0, beta1 = np.linalg.lstsq(Xmat, y, rcond=None)[0]
    kappa = -beta1
    theta = beta0 / kappa if kappa != 0 else v_ts.mean()

    # --- xi：残差法 ---
    eps   = y - (beta0 + beta1 * X)
    xi    = np.sqrt(np.var(eps) * dt / v_ts.mean())

    # --- rho：相关 ---
    dW1 = (r - (mu_daily - 0.5 * v_ts) * dt) / np.sqrt(v_ts * dt)
    # 对齐长度
    v_ts_mid = v_ts.iloc[:-1]
    eps_mid  = eps
    dW2 = eps_mid / (xi * np.sqrt(v_ts_mid) )
    rho = np.corrcoef(dW1.iloc[1:], dW2)[0, 1]   # 两系列时间对齐

    # --- 模拟 ---
    spot = grp["Close"].iloc[-1]
    paths = simulate_heston_paths(
        spot, mu_daily, v0, kappa, theta, xi, rho,
        days=days_to_sim, n_paths=n_paths, dt=dt
    )

    results[tic] = dict(mu=mu_daily, v0=v0, kappa=kappa,
                        theta=theta, xi=xi, rho=rho, paths=paths)

    print(f"[{tic}] μ={mu_daily:.6f}  v0={v0:.4e}  κ={kappa:.2f}  θ={theta:.4e}  ξ={xi:.3f}  ρ={rho:.3f}")

# ------------------------------------------------------------
# 4 保存均值路径
# ------------------------------------------------------------
mean_df = pd.DataFrame({tic: res["paths"].mean(axis=0) for tic, res in results.items()})
mean_df.index.name = "Step"
mean_df.to_csv("Heston_mean_paths_one_year.csv")
print("\n均值路径已保存：Heston_mean_paths_one_year.csv")


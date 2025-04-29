import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# === 基本設定 ===
spot_dict = {
    "EFA": 84.640,
    "GLD": 303.950,
    "NFLX": 1046.090,
    "NOW": 810.620,
    "NVDA": 102.208,
    "PANW": 167.795,
    "PLTR": 100.407,
    "TSLA": 251.770,
    "QQQ": 454.600,
    "VISA": 333.820,
    "XLE": 81.065,
    "XLF": 47.675,
}

stock_list = list(spot_dict.keys())

# === 模擬參數 ===
T = 252  # 天數
n_paths = 1000  # 模擬路數
initial_wealth = 100

# Heston參數 (示範，請換成你真正calibrate出來的)
v0_dict = {k: 0.04 for k in stock_list}  # initial variance
kappa_dict = {k: 1.5 for k in stock_list}
theta_dict = {k: 0.04 for k in stock_list}  # long-term variance
sigma_v_dict = {k: 0.3 for k in stock_list}  # volatility of variance
rho_dict = {k: -0.5 for k in stock_list}

# === 原本給的 optimized weights ===
original_weights = np.array(
    [
        -0.2721,
        1.0000,
        0.2209,
        -0.0856,
        0.1279,
        0.0957,
        0.1917,
        0.0651,
        -1.0000,
        0.3447,
        -0.2493,
        0.5610,
    ]
)


# === Heston 模型模擬器 ===
def simulate_heston(mu=0.05):
    spot_array = np.array([spot_dict[s] for s in stock_list])
    v0_array = np.array([v0_dict[s] for s in stock_list])
    kappa_array = np.array([kappa_dict[s] for s in stock_list])
    theta_array = np.array([theta_dict[s] for s in stock_list])
    sigma_v_array = np.array([sigma_v_dict[s] for s in stock_list])
    rho_array = np.array([rho_dict[s] for s in stock_list])

    dt = 1 / 252
    paths = np.zeros((T, n_paths, len(stock_list)))
    variances = np.zeros((T, n_paths, len(stock_list)))
    paths[0] = np.tile(spot_array, (n_paths, 1))
    variances[0] = np.tile(v0_array, (n_paths, 1))

    for t in range(1, T):
        z1 = np.random.randn(n_paths, len(stock_list))
        z2 = np.random.randn(n_paths, len(stock_list))
        z2 = rho_array * z1 + np.sqrt(1 - rho_array**2) * z2  # correlate z2 with z1

        v_prev = np.maximum(variances[t - 1], 1e-8)  # 保持正的variance

        variances[t] = (
            variances[t - 1]
            + kappa_array * (theta_array - v_prev) * dt
            + sigma_v_array * np.sqrt(v_prev * dt) * z2
        )
        variances[t] = np.maximum(variances[t], 1e-8)  # 保持正的variance

        paths[t] = paths[t - 1] * np.exp(
            (mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * z1
        )
    return paths


# === 投資組合價值計算器 ===
def portfolio_final_values(weights, paths):
    final_prices = paths[-1]  # (n_paths, n_stocks)
    portfolio_values = np.dot(final_prices, weights)
    return portfolio_values


# === Sharpe 計算器 ===
def calculate_sharpe(portfolio_values):
    returns = (portfolio_values - initial_wealth) / initial_wealth
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / std_return if std_return != 0 else -np.inf
    return sharpe


# === 最佳化 Sharpe ===
def optimize_weights(paths):
    n_stocks = paths.shape[2]

    def neg_sharpe(w):
        w = np.array(w)
        pf = portfolio_final_values(w, paths)
        return -calculate_sharpe(pf)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(-2, 2)] * n_stocks
    w0 = np.ones(n_stocks) / n_stocks

    res = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return res.x


# === 主程式 ===
def main():
    # 1. 模擬資料
    paths = simulate_heston()

    # 2. 原本權重
    pf_original = portfolio_final_values(original_weights, paths)
    original_sharpe = calculate_sharpe(pf_original)

    # 3. 找最佳權重
    best_weights = optimize_weights(paths)
    pf_best = portfolio_final_values(best_weights, paths)
    best_sharpe = calculate_sharpe(pf_best)

    # 4. 畫圖比較
    x = np.arange(len(stock_list))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, original_weights, width, label="Original")
    ax.bar(x + width / 2, best_weights, width, label="Optimized")
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights Comparison (Heston Model)")
    ax.set_xticks(x)
    ax.set_xticklabels(stock_list, rotation=45)
    ax.legend()
    plt.grid()
    plt.show()

    print(f"Original Sharpe Ratio: {original_sharpe:.3f}")
    print(f"Optimized Sharpe Ratio: {best_sharpe:.3f}")


if __name__ == "__main__":
    main()

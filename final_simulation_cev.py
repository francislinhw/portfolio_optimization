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

# === CEV 模型參數 ===
cev_params = {
    "EFA": {"sigma": 0.1817, "gamma": 0.5000015},
    "GLD": {"sigma": 0.1886, "gamma": 0.5000038},
    "NFLX": {"sigma": 0.3797, "gamma": 0.5},
    "NOW": {"sigma": 0.4149, "gamma": 0.5000001},
    "NVDA": {"sigma": 0.4869, "gamma": 0.4999622},
    "PANW": {"sigma": 0.4538, "gamma": 0.4999578},
    "PLTR": {"sigma": 0.9766, "gamma": 0.5000202},
    "TSLA": {"sigma": 0.6865, "gamma": 0.5000175},
    "QQQ": {"sigma": 0.2696, "gamma": 0.499991},
    "VISA": {"sigma": 0.293, "gamma": 0.5},
    "XLE": {"sigma": 0.3044, "gamma": 0.5000018},
    "XLF": {"sigma": 0.2544, "gamma": 0.4999052},
}

# === 原本權重 ===
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


# === CEV 模擬器 ===
def simulate_cev(mu=0.05):
    spot_array = np.array([spot_dict[s] for s in stock_list])
    sigma_array = np.array([cev_params[s]["sigma"] for s in stock_list])
    gamma_array = np.array([cev_params[s]["gamma"] for s in stock_list])

    paths = np.zeros((T, n_paths, len(stock_list)))
    paths[0] = np.tile(spot_array, (n_paths, 1))
    dt = 1 / 252

    for t in range(1, T):
        z = np.random.randn(n_paths, len(stock_list))
        prev = paths[t - 1]
        diffusion = (
            sigma_array * (np.maximum(prev, 1e-6) ** gamma_array) * np.sqrt(dt) * z
        )
        drift = mu * prev * dt
        paths[t] = prev + drift + diffusion
    return paths

# === 补充：可视化模拟路径 ===
def plot_simulated_paths(paths, stock_names, num_paths_to_plot=10):
    T, n_paths, n_stocks = paths.shape
    time_axis = np.arange(T)

    for i, stock in enumerate(stock_names):
        plt.figure(figsize=(10, 4))
        for j in range(min(num_paths_to_plot, n_paths)):
            plt.plot(time_axis, paths[:, j, i], alpha=0.6)
        plt.title(f"CEV Simulated Paths for {stock}")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === 計算投資組合價值 ===
def portfolio_final_values(weights, paths):
    final_prices = paths[-1]
    portfolio_values = np.dot(final_prices, weights)
    return portfolio_values


# === 計算 Sharpe Ratio ===
def calculate_sharpe(portfolio_values):
    returns = (portfolio_values - initial_wealth) / initial_wealth
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return if std_return != 0 else -np.inf


# === 最佳化權重 ===
def optimize_weights(paths):
    n_stocks = paths.shape[2]

    def neg_sharpe(w):
        pf = portfolio_final_values(w, paths)
        return -calculate_sharpe(pf)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(-2, 2)] * n_stocks
    w0 = np.ones(n_stocks) / n_stocks

    result = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result.x


# === 主程式 ===
def main():
    paths = simulate_cev()

    # 补充：可视化部分模拟路径（ 6 个样本的前 10 条路径）
    plot_simulated_paths(paths, ["QQQ", "EFA", "XLF", "XLE", "NVDA", "NFLX"], num_paths_to_plot=10)
    
    pf_original = portfolio_final_values(original_weights, paths)
    original_sharpe = calculate_sharpe(pf_original)

    best_weights = optimize_weights(paths)
    pf_best = portfolio_final_values(best_weights, paths)
    best_sharpe = calculate_sharpe(pf_best)

    x = np.arange(len(stock_list))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, original_weights, width, label="Original")
    ax.bar(x + width / 2, best_weights, width, label="Optimized")
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights Comparison (CEV Model)")
    ax.set_xticks(x)
    ax.set_xticklabels(stock_list, rotation=45)
    ax.legend()
    plt.grid()
    plt.show()

    print(f"Original Sharpe Ratio: {original_sharpe:.3f}")
    print(f"Optimized Sharpe Ratio: {best_sharpe:.3f}")


if __name__ == "__main__":
    main()

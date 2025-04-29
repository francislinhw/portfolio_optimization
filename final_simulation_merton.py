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

# === Merton 模型參數 (從你提供的表格中整理) ===
merton_params = {
    "EFA": {"sigma": 0.16998, "lambda": 0.09391, "mu_J": -0.27315, "sigma_J": 0.20928},
    "GLD": {"sigma": 0.18336, "lambda": 0.09859, "mu_J": 0.00018, "sigma_J": 0.19906},
    "NFLX": {"sigma": 0.37387, "lambda": 0.11684, "mu_J": -0.00237, "sigma_J": 0.21130},
    "NOW": {"sigma": 0.40976, "lambda": 0.12380, "mu_J": -0.00165, "sigma_J": 0.21553},
    "NVDA": {"sigma": 0.35937, "lambda": 0.46111, "mu_J": -0.95320, "sigma_J": 0.65101},
    "PANW": {"sigma": 0.35390, "lambda": 0.54719, "mu_J": -0.54821, "sigma_J": 0.50462},
    "PLTR": {"sigma": 0.98255, "lambda": 0.0001, "mu_J": 0.00287, "sigma_J": 0.18688},
    "TSLA": {"sigma": 0.63195, "lambda": 0.27587, "mu_J": -0.56724, "sigma_J": 0.17196},
    "QQQ": {"sigma": 0.23624, "lambda": 0.10946, "mu_J": -0.84471, "sigma_J": 0.04582},
    "VISA": {"sigma": 0.31946, "lambda": 0.08313, "mu_J": 1.0, "sigma_J": 0.24063},
    "XLE": {"sigma": 0.21733, "lambda": 0.86686, "mu_J": -0.29525, "sigma_J": 0.0001},
    "XLF": {"sigma": 0.22995, "lambda": 0.11708, "mu_J": -0.55130, "sigma_J": 0.04873},
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


# === Merton 模擬器 ===
def simulate_merton(mu=0.05):
    spot_array = np.array([spot_dict[s] for s in stock_list])
    sigma_array = np.array([merton_params[s]["sigma"] for s in stock_list])
    lambda_array = np.array([merton_params[s]["lambda"] for s in stock_list])
    mu_J_array = np.array([merton_params[s]["mu_J"] for s in stock_list])
    sigma_J_array = np.array([merton_params[s]["sigma_J"] for s in stock_list])

    paths = np.zeros((T, n_paths, len(stock_list)))
    paths[0] = np.tile(spot_array, (n_paths, 1))
    dt = 1 / 252

    for t in range(1, T):
        z = np.random.randn(n_paths, len(stock_list))
        n_jumps = np.random.poisson(lambda_array * dt, size=(n_paths, len(stock_list)))
        jump_sizes = (
            np.random.normal(mu_J_array, sigma_J_array, size=(n_paths, len(stock_list)))
            * n_jumps
        )
        drift = (mu - 0.5 * sigma_array**2) * dt
        diffusion = sigma_array * np.sqrt(dt) * z
        jump = jump_sizes
        paths[t] = paths[t - 1] * np.exp(drift + diffusion + jump)

    return paths


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
    paths = simulate_merton()

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
    ax.set_title("Portfolio Weights Comparison (Merton Model)")
    ax.set_xticks(x)
    ax.set_xticklabels(stock_list, rotation=45)
    ax.legend()
    plt.grid()
    plt.show()

    print(f"Original Sharpe Ratio: {original_sharpe:.3f}")
    print(f"Optimized Sharpe Ratio: {best_sharpe:.3f}")


if __name__ == "__main__":
    main()

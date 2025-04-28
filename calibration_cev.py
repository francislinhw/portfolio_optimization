import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from src.cev.pricer import price_cev

import matplotlib.pyplot as plt


def find_sigma_from_market(market_data, spot):
    market_data["moneyness"] = np.abs(market_data["strike"] - spot)
    atm_row = market_data.loc[market_data["moneyness"].idxmin()]
    sigma_market = atm_row["market_iv"]
    print(
        f"Selected ATM strike: {atm_row['strike']}, Market IV (sigma): {sigma_market*100:.2f}%"
    )
    return sigma_market


# 畫 gamma -> loss 曲線
def plot_loss_vs_gamma(market_data, spot, sigma, r=0.05):
    gamma_grid = np.linspace(0.01, 1.0, 50)
    losses = []

    for gamma in gamma_grid:
        loss = calibration_loss(gamma, market_data, spot, sigma, r)
        losses.append(loss)

    plt.figure(figsize=(8, 5))
    plt.plot(gamma_grid, losses, marker="o")
    plt.xlabel("Gamma")
    plt.ylabel("Loss")
    plt.title("Loss vs Gamma")
    plt.grid()
    plt.show()


# 畫市場IV vs 模型IV
def plot_vol_smile(market_data, spot, sigma, gamma_best, r=0.05):
    strikes = []
    market_ivs = []
    model_ivs = []

    for idx, row in market_data.iterrows():
        strikes.append(row["strike"])
        market_ivs.append(row["market_iv"])

        model_price = price_cev(
            spot, row["strike"], row["expiry_date"], r=r, sigma=sigma, gamma=gamma_best
        )["call"]
        model_iv = implied_volatility_from_price(
            spot, row["strike"], row["expiry_date"], r, model_price
        )
        model_ivs.append(model_iv)

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, np.array(market_ivs) * 100, "o-", label="Market IV")
    plt.plot(strikes, np.array(model_ivs) * 100, "s--", label="Model IV (CEV)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.title("Volatility Smile Comparison")
    plt.legend()
    plt.grid()
    plt.show()


# 1. implied volatility 反推函數
def implied_volatility_from_price(
    spot, strike, maturity_date, r, option_price, option_type="call"
):
    today = pd.Timestamp.today()
    T = (maturity_date - today).days / 365

    def bs_price(sigma):
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        else:
            return strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def objective(sigma):
        return bs_price(sigma) - option_price

    try:
        implied_vol = brentq(objective, 1e-6, 5, xtol=1e-6)
    except ValueError:
        implied_vol = np.nan
    return implied_vol


# 2. loss function
def calibration_loss(gamma, market_data, spot, sigma, r=0.05):
    loss = 0
    for idx, row in market_data.iterrows():
        try:
            model_price = price_cev(
                spot, row["strike"], row["expiry_date"], r=r, sigma=sigma, gamma=gamma
            )["call"]
            model_iv = implied_volatility_from_price(
                spot, row["strike"], row["expiry_date"], r, model_price
            )
            if not np.isnan(model_iv):
                loss += (model_iv - row["market_iv"]) ** 2
        except Exception as e:
            print(f"Warning: error at idx {idx}: {e}")
            continue
    return loss


# 3. calibration主程序
def calibrate_gamma(market_data, spot, sigma, r=0.05):
    result = minimize(
        lambda gamma: calibration_loss(gamma, market_data, spot, sigma, r),
        x0=[0.5],
        bounds=[(0.00, 1.0)],
        method="L-BFGS-B",
        options={"maxiter": 1000},  # 多跑一點步數
    )
    return result.x[0]


# 3. calibration主程序
def calibrate_gamma(market_data, spot, sigma, r=0.05):
    result = minimize(
        lambda gamma: calibration_loss(gamma, market_data, spot, sigma, r),
        x0=[0.5],
        bounds=[(0.01, 1.0)],
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-13},
    )
    return result.x[0]


market_data = pd.DataFrame(
    [
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 71.0,
            "market_iv": 27.00 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 71.5,
            "market_iv": 26.80 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 72.0,
            "market_iv": 26.60 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 72.5,
            "market_iv": 26.40 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 73.0,
            "market_iv": 26.20 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 73.5,
            "market_iv": 26.00 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 74.0,
            "market_iv": 25.80 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 74.5,
            "market_iv": 25.60 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 75.0,
            "market_iv": 25.40 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 75.5,
            "market_iv": 25.20 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 76.0,
            "market_iv": 25.00 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 76.5,
            "market_iv": 24.80 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 77.0,
            "market_iv": 24.60 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 77.5,
            "market_iv": 24.40 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 78.0,
            "market_iv": 24.20 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 78.5,
            "market_iv": 24.00 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 79.0,
            "market_iv": 23.80 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 79.5,
            "market_iv": 23.60 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 80.0,
            "market_iv": 23.40 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 80.5,
            "market_iv": 23.20 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 81.0,
            "market_iv": 23.00 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 81.5,
            "market_iv": 22.80 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2026-03-20"),
            "strike": 82.0,
            "market_iv": 22.60 / 100,
        },
    ]
)

# 你可以這樣用：
sigma_market = find_sigma_from_market(market_data, spot=78.29)
best_gamma = calibrate_gamma(market_data, spot=78.29, sigma=sigma_market)
print("Best gamma:", best_gamma)

# 畫 Vol Smile
plot_vol_smile(market_data, spot=78.29, sigma=sigma_market, gamma_best=best_gamma)

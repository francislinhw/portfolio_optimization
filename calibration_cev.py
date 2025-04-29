import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from src.cev.pricer import price_cev
from data.volatility.EFA import efa_market_data
from data.volatility.GLD import gld_market_data
from data.volatility.NFLX import nflx_market_data
from data.volatility.NOW import now_market_data
from data.volatility.NVDA import nvda_market_data
from data.volatility.PANW import panw_market_data
from data.volatility.PLTR import pltr_market_data
from data.volatility.TSLA import tsla_market_data
from data.volatility.QQQ import qqq_market_data
from data.volatility.VISA import visa_market_data
from data.volatility.XLE import xle_market_data
from data.volatility.XLF import xlf_market_data
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


def mainCalibration(market_data, spot, r=0.05):
    sigma_market = find_sigma_from_market(market_data, spot=spot)
    best_gamma = calibrate_gamma(market_data, spot=spot, sigma=sigma_market)
    print("Best gamma:", best_gamma)

    # 畫 Vol Smile
    plot_vol_smile(market_data, spot=spot, sigma=sigma_market, gamma_best=best_gamma)
    result = {
        "gamma": best_gamma,
        "sigma": sigma_market,
    }
    return result


if __name__ == "__main__":
    all_result = []
    stock_list = [
        "EFA",
        "GLD",
        "NFLX",
        "NOW",
        "NVDA",
        "PANW",
        "PLTR",
        "TSLA",
        "QQQ",
        "VISA",
        "XLE",
        "XLF",
    ]
    market_data_dict = {
        "EFA": efa_market_data,
        "GLD": gld_market_data,
        "NFLX": nflx_market_data,
        "NOW": now_market_data,
        "NVDA": nvda_market_data,
        "PANW": panw_market_data,
        "PLTR": pltr_market_data,
        "TSLA": tsla_market_data,
        "QQQ": qqq_market_data,
        "VISA": visa_market_data,
        "XLE": xle_market_data,
        "XLF": xlf_market_data,
    }
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
    for stock in stock_list:
        market_data = market_data_dict[stock]
        spot = spot_dict[stock]
        result = mainCalibration(market_data, spot=spot, r=0.05)
        all_result.append({"stock": stock, "result": result})
    all_result_df = pd.DataFrame(all_result)
    all_result_df.to_csv("calibration_cev_result.csv", index=False)

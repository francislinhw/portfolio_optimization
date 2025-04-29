import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
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


# Merton Jump-Diffusion pricing function
def merton_jump_price(
    spot, strike, maturity, r, mu, sigma, lam, muj, sigmaj, option_type="call"
):
    price = 0
    for n in range(50):  # summation truncation
        poisson_prob = (
            np.exp(-lam * maturity) * (lam * maturity) ** n / np.math.factorial(n)
        )
        sigma_n = np.sqrt(sigma**2 + (n * sigmaj**2) / maturity)
        r_n = r - lam * (np.exp(muj + 0.5 * sigmaj**2) - 1) + (n * muj) / maturity

        d1 = (np.log(spot / strike) + (r_n + 0.5 * sigma_n**2) * maturity) / (
            sigma_n * np.sqrt(maturity)
        )
        d2 = d1 - sigma_n * np.sqrt(maturity)

        if option_type == "call":
            bs = spot * norm.cdf(d1) - strike * np.exp(-r_n * maturity) * norm.cdf(d2)
        else:
            bs = strike * np.exp(-r_n * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        price += poisson_prob * bs
    return price


# Black-Scholes implied volatility extraction
def implied_volatility_from_price(
    spot, strike, maturity, r, option_price, option_type="call"
):
    def bs_price(sigma):
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * maturity) / (
            sigma * np.sqrt(maturity)
        )
        d2 = d1 - sigma * np.sqrt(maturity)
        if option_type == "call":
            return spot * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)
        else:
            return strike * np.exp(-r * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def objective(sigma):
        return bs_price(sigma) - option_price

    try:
        implied_vol = brentq(objective, 1e-6, 5)
    except ValueError:
        implied_vol = np.nan
    return implied_vol


# Calibration loss function for Merton
def calibration_loss_merton(params, market_data, spot, r):
    mu, sigma, lam, muj, sigmaj = params
    loss = 0
    today = pd.Timestamp.today()

    for _, row in market_data.iterrows():
        maturity = (row["expiry_date"] - today).days / 365
        if maturity <= 0:
            continue
        model_price = merton_jump_price(
            spot, row["strike"], maturity, r, mu, sigma, lam, muj, sigmaj
        )
        model_iv = implied_volatility_from_price(
            spot, row["strike"], maturity, r, model_price
        )
        if not np.isnan(model_iv):
            loss += (model_iv - row["market_iv"]) ** 2
    return loss


# Calibration main function
def calibrate_merton(market_data, spot, r=0.05):
    x0 = [
        0.0,
        0.2,
        0.1,
        0.0,
        0.2,
    ]  # initial guess for [mu, sigma, lambda, mu_j, sigma_j]
    bounds = [
        (-1, 1),  # mu
        (1e-4, 2.0),  # sigma
        (1e-4, 2.0),  # lambda
        (-1, 1),  # mu_j
        (1e-4, 2.0),  # sigma_j
    ]
    result = minimize(
        lambda params: calibration_loss_merton(params, market_data, spot, r),
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 3000},
    )
    return result.x


# Optional: Plotting function
def plot_merton_vol_smile(market_data, spot, best_params, r=0.05):
    mu, sigma, lam, muj, sigmaj = best_params
    strikes = []
    market_ivs = []
    model_ivs = []

    today = pd.Timestamp.today()
    for _, row in market_data.iterrows():
        strikes.append(row["strike"])
        market_ivs.append(row["market_iv"])
        maturity = (row["expiry_date"] - today).days / 365
        if maturity <= 0:
            model_ivs.append(np.nan)
            continue
        price = merton_jump_price(
            spot, row["strike"], maturity, r, mu, sigma, lam, muj, sigmaj
        )
        iv = implied_volatility_from_price(spot, row["strike"], maturity, r, price)
        model_ivs.append(iv)

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, np.array(market_ivs) * 100, "o-", label="Market IV")
    plt.plot(strikes, np.array(model_ivs) * 100, "s--", label="Model IV (Merton)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.title("Volatility Smile - Merton Calibration")
    plt.legend()
    plt.grid()
    plt.show()


def mainCalibration(market_data, spot=78.29, r=0.05):

    def find_sigma_from_market(market_data, spot):
        market_data["moneyness"] = np.abs(market_data["strike"] - spot)
        atm_row = market_data.loc[market_data["moneyness"].idxmin()]
        sigma_market = atm_row["market_iv"]
        print(
            f"Selected ATM strike: {atm_row['strike']}, Market IV (sigma): {sigma_market*100:.2f}%"
        )
        return sigma_market

    sigma_market = find_sigma_from_market(market_data, spot=spot)

    best_params = calibrate_merton(market_data, spot=spot, r=r)

    print("Best Merton Parameters: μ, σ, λ, μ_J, σ_J")
    print(best_params)

    plot_merton_vol_smile(
        market_data,
        spot=spot,
        best_params=best_params,
        r=r,
    )

    result = {
        "mu": best_params[0],
        "sigma": best_params[1],
        "lambda": best_params[2],
        "mu_J": best_params[3],
        "sigma_J": best_params[4],
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
        print(f"=== Calibrating {stock} ===")
        market_data = market_data_dict[stock]
        spot = spot_dict[stock]
        result = mainCalibration(market_data, spot=spot, r=0.05)
        all_result.append({"stock": stock, "result": result})

    all_result_df = pd.DataFrame(all_result)
    all_result_df.to_csv("merton_calibration_result.csv", index=False)

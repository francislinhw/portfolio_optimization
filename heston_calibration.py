import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import cmath
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


def plot_heston_vol_smile(market_data, spot, best_params, r=0.05, sigma_market=0.2):
    v0, kappa, theta, rho = best_params
    sigma = sigma_market
    strikes = []
    market_ivs = []
    model_ivs = []

    today = pd.Timestamp.today()

    for idx, row in market_data.iterrows():
        strikes.append(row["strike"])
        market_ivs.append(row["market_iv"])

        maturity = (row["expiry_date"] - today).days / 365
        if maturity <= 0:
            model_ivs.append(np.nan)
            continue

        model_price = heston_price(
            spot, row["strike"], maturity, r, v0, kappa, theta, sigma, rho
        )
        model_iv = implied_volatility_from_price(
            spot, row["strike"], maturity, r, model_price
        )
        model_ivs.append(model_iv)

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, np.array(market_ivs) * 100, "o-", label="Market IV")
    plt.plot(strikes, np.array(model_ivs) * 100, "s--", label="Model IV (Heston)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.title("Volatility Smile - Heston Calibration")
    plt.legend()
    plt.grid()
    plt.show()


# Heston closed-form pricing function
def heston_price(
    spot, strike, maturity, r, v0, kappa, theta, sigma, rho, option_type="call"
):
    def char_func(phi, Pnum):
        a = kappa * theta
        u = 0.5 if Pnum == 1 else -0.5
        b = kappa - rho * sigma if Pnum == 1 else kappa
        d = np.sqrt(
            (rho * sigma * 1j * phi - b) ** 2 - sigma**2 * (2 * u * 1j * phi - phi**2)
        )
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
        C = r * 1j * phi * maturity + a / sigma**2 * (
            (b - rho * sigma * 1j * phi + d) * maturity
            - 2 * np.log((1 - g * np.exp(d * maturity)) / (1 - g))
        )
        D = (
            (b - rho * sigma * 1j * phi + d)
            / sigma**2
            * ((1 - np.exp(d * maturity)) / (1 - g * np.exp(d * maturity)))
        )
        return np.exp(C + D * v0 + 1j * phi * np.log(spot))

    def integrand(phi, Pnum):
        cf = char_func(phi, Pnum)
        return np.real(np.exp(-1j * phi * np.log(strike)) * cf / (1j * phi))

    integral_P1 = np.trapz(
        [integrand(phi, 1) for phi in np.linspace(1e-5, 100, 1000)],
        np.linspace(1e-5, 100, 1000),
    )
    integral_P2 = np.trapz(
        [integrand(phi, 2) for phi in np.linspace(1e-5, 100, 1000)],
        np.linspace(1e-5, 100, 1000),
    )

    P1 = 0.5 + 1 / np.pi * integral_P1
    P2 = 0.5 + 1 / np.pi * integral_P2

    call_price = spot * P1 - strike * np.exp(-r * maturity) * P2
    if option_type == "call":
        return call_price
    else:
        return call_price - spot + strike * np.exp(-r * maturity)


# Implied volatility反推
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


# Calibration loss function
def calibration_loss_heston(params, market_data, spot, r, sigma_market):
    v0, kappa, theta, rho = params
    sigma = sigma_market
    loss = 0
    today = pd.Timestamp.today()

    for idx, row in market_data.iterrows():
        maturity = (row["expiry_date"] - today).days / 365
        if maturity <= 0:
            continue
        model_price = heston_price(
            spot, row["strike"], maturity, r, v0, kappa, theta, sigma, rho
        )
        model_iv = implied_volatility_from_price(
            spot, row["strike"], maturity, r, model_price
        )

        if not np.isnan(model_iv):
            loss += (model_iv - row["market_iv"]) ** 2
    return loss


# Calibration main function
def calibrate_heston(market_data, spot, r=0.05, sigma_market=0.2):
    x0 = [0.05, 1.0, 0.05, -0.5]  # v0, kappa, theta, sigma, rho
    bounds = [
        (1e-6, 2.0),  # v0
        (1e-3, 5.0),  # kappa
        (1e-6, 2.0),  # theta
        (-0.999, 0.0),  # rho
    ]

    result = minimize(
        lambda params: calibration_loss_heston(
            params, market_data, spot, r, sigma_market
        ),
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 5000},
    )

    return result.x  # return best (v0, kappa, theta, sigma, rho)


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
    best_params = calibrate_heston(market_data, spot, r, sigma_market)
    print("Best Heston Parameters: kappa, theta, rho")
    print(best_params)
    plot_heston_vol_smile(
        market_data,
        spot=spot,
        best_params=best_params,
        r=r,
        sigma_market=sigma_market,
    )
    result = {
        "v0": sigma_market,
        "kappa": best_params[0],
        "theta": best_params[1],
        "volOfVol": best_params[2],
        "rho": best_params[3],
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
    all_result_df.to_csv("heston_calibration_result.csv", index=False)

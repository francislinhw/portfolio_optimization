import numpy as np
from src.market_setup import price_option_from_paths
import pandas as pd


def simulate_cev_paths(S0, r, sigma, gamma, T, N=252, n_paths=10000):
    dt = T / N
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0

    for t in range(1, N + 1):
        Z = np.random.normal(size=n_paths)
        dS = (
            r * S[:, t - 1] * dt
            + sigma * np.power(np.maximum(S[:, t - 1], 1e-6), gamma) * np.sqrt(dt) * Z
        )
        S[:, t] = np.maximum(S[:, t - 1] + dS, 1e-6)  # 防止負數或爆炸
    return S


def price_cev(spot, strike, maturity_date, r=0.05, sigma=0.2, gamma=1):

    today = pd.Timestamp.today()
    T = (maturity_date - today).days / 365  # year

    S_paths = simulate_cev_paths(S0=spot, r=r, sigma=sigma, gamma=gamma, T=T)
    call_price = price_option_from_paths(S_paths, strike, r, T, option_type="call")
    put_price = price_option_from_paths(S_paths, strike, r, T, option_type="put")

    return {"call": call_price, "put": put_price}

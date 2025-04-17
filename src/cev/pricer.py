import numpy as np
from src.market_setup import price_option_from_paths


def simulate_cev_paths(S0, r, sigma, gamma, T, dt=1 / 252, n_paths=10000):
    steps = int(T / dt)
    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):
        Z = np.random.normal(size=n_paths)
        dS = r * S[:, t - 1] * dt + sigma * (S[:, t - 1] ** gamma) * np.sqrt(dt) * Z
        S[:, t] = np.maximum(S[:, t - 1] + dS, 1e-8)  # negative price is not allowed

    return S


def price_cev(spot, strike, maturity_date, r=0.05, sigma=0.2, gamma=1):
    from datetime import date

    today = date.today()
    T = (maturity_date - today).days / 365  # year

    S_paths = simulate_cev_paths(S0=spot, r=r, sigma=sigma, gamma=gamma, T=T)
    call_price = price_option_from_paths(S_paths, strike, r, T, option_type="call")
    put_price = price_option_from_paths(S_paths, strike, r, T, option_type="put")

    return {"call": call_price, "put": put_price}

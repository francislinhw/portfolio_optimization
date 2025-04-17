import numpy as np
from src.market_setup import price_option_from_paths


def simulate_merton_paths(
    S0, mu, sigma, T, dt, jump_lambda, jump_mu, jump_sigma, n_paths=10000
):
    steps = int(T / dt)
    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):
        Z = np.random.normal(size=n_paths)
        J = np.random.poisson(jump_lambda * dt, n_paths)
        Y = np.random.normal(jump_mu, jump_sigma, n_paths) * J
        dS = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + Y
        S[:, t] = S[:, t - 1] * np.exp(dS)

    return S


def price_merton_mc(
    spot,
    strike,
    maturity_date,
    r=0.05,
    sigma=0.2,
    lambda_jump=0.75,
    mean_jump=-0.1,
    jump_vol=0.3,
    n_paths=100000,
):
    from datetime import date

    today = date.today()
    T = (maturity_date - today).days / 365

    S_paths = simulate_merton_paths(
        S0=spot,
        mu=r,
        sigma=sigma,
        T=T,
        dt=1 / 252,
        jump_lambda=lambda_jump,
        jump_mu=mean_jump,
        jump_sigma=jump_vol,
        n_paths=n_paths,
    )

    call_price = price_option_from_paths(S_paths, strike, r, T, option_type="call")
    put_price = price_option_from_paths(S_paths, strike, r, T, option_type="put")

    return {"call": call_price, "put": put_price}

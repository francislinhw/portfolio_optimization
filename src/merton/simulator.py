import numpy as np


def simulate_merton_paths(
    spot, mu, sigma, lamb, jump_mu, jump_sigma, days, n_paths, dt=1 / 252
):
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = spot
    for t in range(1, days + 1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        J = np.random.poisson(lamb * dt, n_paths)
        jumps = np.random.normal(jump_mu, jump_sigma, n_paths) * J
        paths[:, t] = paths[:, t - 1] * (1 + mu * dt + sigma * dW + jumps)
    return paths

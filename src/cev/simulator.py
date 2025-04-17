import numpy as np


def simulate_cev_paths(spot, mu, sigma, gamma, days, n_paths, dt=1 / 252):
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = spot
    for t in range(1, days + 1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        dS = mu * paths[:, t - 1] * dt + sigma * np.power(paths[:, t - 1], gamma) * dW
        paths[:, t] = np.maximum(paths[:, t - 1] + dS, 1e-8)
    return paths

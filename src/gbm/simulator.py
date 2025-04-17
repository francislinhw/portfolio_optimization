import numpy as np


def simulate_gbm_paths(spot, mu, sigma, days, n_paths, dt=1 / 252):
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = spot
    for t in range(1, days + 1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:, t] = paths[:, t - 1] * (1 + mu * dt + sigma * dW)
    return paths

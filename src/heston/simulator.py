import numpy as np


def simulate_heston_paths(
    spot, mu, v0, kappa, theta, xi, rho, days, n_paths, dt=1 / 252
):
    paths = np.zeros((n_paths, days + 1))
    paths[:, 0] = spot
    v = np.full(n_paths, v0)
    for t in range(1, days + 1):
        dW1 = np.random.normal(0, np.sqrt(dt), n_paths)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(
            0, np.sqrt(dt), n_paths
        )
        v = np.maximum(v + kappa * (theta - v) * dt + xi * np.sqrt(v) * dW2, 1e-8)
        dS = mu * paths[:, t - 1] * dt + np.sqrt(v) * paths[:, t - 1] * dW1
        paths[:, t] = paths[:, t - 1] + dS
    return paths

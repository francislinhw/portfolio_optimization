import numpy as np


def simulate_correlated_gbm_paths(
    spot_vec, mu_vec, sigma_vec, corr_matrix, days, n_paths, dt=1 / 252
):
    n_assets = len(spot_vec)
    L = np.linalg.cholesky(corr_matrix)
    paths = np.zeros((n_assets, n_paths, days + 1))
    paths[:, :, 0] = np.array(spot_vec)[:, None]

    for t in range(1, days + 1):
        Z = np.random.normal(0, 1, (n_assets, n_paths))
        Z_corr = L @ Z
        drift = (mu_vec[:, None] - 0.5 * sigma_vec[:, None] ** 2) * dt
        diffusion = sigma_vec[:, None] * np.sqrt(dt) * Z_corr
        paths[:, :, t] = paths[:, :, t - 1] * np.exp(drift + diffusion)

    return paths


def simulate_correlated_merton_paths(
    spot_vec,
    mu_vec,
    sigma_vec,
    lamb_vec,
    jump_mu_vec,
    jump_sigma_vec,
    corr_matrix,
    days,
    n_paths,
    dt=1 / 252,
):
    n_assets = len(spot_vec)
    L = np.linalg.cholesky(corr_matrix)
    paths = np.zeros((n_assets, n_paths, days + 1))
    paths[:, :, 0] = np.array(spot_vec)[:, None]

    for t in range(1, days + 1):
        Z = np.random.normal(0, 1, (n_assets, n_paths))
        Z_corr = L @ Z
        J = np.random.poisson(lamb_vec[:, None] * dt, (n_assets, n_paths))
        jumps = (
            np.random.normal(
                jump_mu_vec[:, None], jump_sigma_vec[:, None], (n_assets, n_paths)
            )
            * J
        )
        drift = (mu_vec[:, None] - 0.5 * sigma_vec[:, None] ** 2) * dt
        diffusion = sigma_vec[:, None] * np.sqrt(dt) * Z_corr
        paths[:, :, t] = paths[:, :, t - 1] * np.exp(drift + diffusion + jumps)

    return paths


def simulate_correlated_cev_paths(
    spot_vec, mu_vec, sigma_vec, gamma_vec, corr_matrix, days, n_paths, dt=1 / 252
):
    n_assets = len(spot_vec)
    L = np.linalg.cholesky(corr_matrix)
    paths = np.zeros((n_assets, n_paths, days + 1))
    paths[:, :, 0] = np.array(spot_vec)[:, None]

    for t in range(1, days + 1):
        Z = np.random.normal(0, 1, (n_assets, n_paths))
        Z_corr = L @ Z
        S_prev = paths[:, :, t - 1]
        drift = mu_vec[:, None] * S_prev * dt
        diffusion = (
            sigma_vec[:, None] * (S_prev ** gamma_vec[:, None]) * np.sqrt(dt) * Z_corr
        )
        paths[:, :, t] = np.maximum(S_prev + drift + diffusion, 1e-8)

    return paths


def simulate_correlated_heston_paths(
    spot_vec,
    mu_vec,
    v0_vec,
    kappa_vec,
    theta_vec,
    xi_vec,
    corr_matrix,
    rho_asset_vol,
    days,
    n_paths,
    dt=1 / 252,
):
    n_assets = len(spot_vec)
    L = np.linalg.cholesky(corr_matrix)
    paths = np.zeros((n_assets, n_paths, days + 1))
    paths[:, :, 0] = np.array(spot_vec)[:, None]
    v = np.tile(v0_vec[:, None], (1, n_paths))

    for t in range(1, days + 1):
        Z1 = np.random.normal(0, 1, (n_assets, n_paths))
        Z2 = np.random.normal(0, 1, (n_assets, n_paths))
        dW1 = L @ Z1
        dW2 = rho_asset_vol[:, None] * dW1 + np.sqrt(
            1 - rho_asset_vol[:, None] ** 2
        ) * (L @ Z2)

        v = np.maximum(
            v
            + kappa_vec[:, None] * (theta_vec[:, None] - v) * dt
            + xi_vec[:, None] * np.sqrt(v) * np.sqrt(dt) * dW2,
            1e-8,
        )
        dS = mu_vec[:, None] * dt + np.sqrt(v) * np.sqrt(dt) * dW1
        paths[:, :, t] = paths[:, :, t - 1] * (1 + dS)

    return paths

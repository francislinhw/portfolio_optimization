import pandas as pd
from datetime import datetime
from src.corr_simulation import (
    simulate_correlated_gbm_paths,
    simulate_correlated_merton_paths,
    simulate_correlated_cev_paths,
    simulate_correlated_heston_paths,
)
import numpy as np

# Example usage for all four models (3 correlated assets)
n_assets = 3
n_paths = 1000
days = 252

# Basic asset parameters
spot_vec = np.array([100, 90, 110])
mu_vec = np.array([0.05, 0.04, 0.06])
sigma_vec = np.array([0.2, 0.25, 0.22])

# Correlation matrix (symmetric and positive-definite)
corr_matrix = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]])

# Merton jump parameters
lamb_vec = np.array([0.7, 0.6, 0.5])
jump_mu_vec = np.array([-0.1, -0.08, -0.05])
jump_sigma_vec = np.array([0.3, 0.25, 0.2])

# CEV parameter
gamma_vec = np.array([0.6, 0.7, 0.8])

# Heston volatility parameters
v0_vec = np.array([0.04, 0.05, 0.03])
kappa_vec = np.array([2.0, 1.8, 1.5])
theta_vec = np.array([0.04, 0.05, 0.03])
xi_vec = np.array([0.3, 0.25, 0.2])
rho_asset_vol = np.array([-0.6, -0.5, -0.4])

# Simulate all four models
gbm_paths = simulate_correlated_gbm_paths(
    spot_vec, mu_vec, sigma_vec, corr_matrix, days, n_paths
)
merton_paths = simulate_correlated_merton_paths(
    spot_vec,
    mu_vec,
    sigma_vec,
    lamb_vec,
    jump_mu_vec,
    jump_sigma_vec,
    corr_matrix,
    days,
    n_paths,
)
cev_paths = simulate_correlated_cev_paths(
    spot_vec, mu_vec, sigma_vec, gamma_vec, corr_matrix, days, n_paths
)
heston_paths = simulate_correlated_heston_paths(
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
)

# Generate correct business day dates
business_dates = pd.bdate_range(start=datetime.today(), periods=days + 1)


# Function to save paths with correct business day index
def save_paths_with_business_days(paths, model_name):
    for i in range(paths.shape[0]):
        df = pd.DataFrame(paths[i].T)
        df.insert(0, "Date", business_dates)
        df.T.to_csv(f"data/{model_name}_asset_{i+1}_business.csv", index=False)


# Save simulated paths with correct business day index
save_paths_with_business_days(gbm_paths, "gbm")
save_paths_with_business_days(merton_paths, "merton")
save_paths_with_business_days(cev_paths, "cev")
save_paths_with_business_days(heston_paths, "heston")

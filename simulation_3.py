import pandas as pd
from datetime import datetime
from src.corr_simulation import (
    simulate_correlated_gbm_paths,
    simulate_correlated_merton_paths,
    simulate_correlated_cev_paths,
    simulate_correlated_heston_paths,
)
import numpy as np
import os
import shutil

corr_df = pd.read_csv('parameter_estimation/corr_matrix.csv', index_col=0)
corr_matrix = corr_df.to_numpy()

parameters_df = pd.read_csv('parameter_estimation/parameters.csv', index_col=0)
assets = parameters_df.index.to_numpy()
parameters_name = parameters_df.columns
for col in parameters_name:
    globals()[f"{col}_vec"] = parameters_df[col].to_numpy()


n_paths = 1000
days = 252

# Simulate all four models
gbm_paths = simulate_correlated_gbm_paths(
    spot_vec, mu_vec, sigma_vec, corr_matrix, days, n_paths
)
merton_paths = simulate_correlated_merton_paths(
    spot_vec,
    mu_vec,
    sigma_vec,
    lambda_vec,
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
    rho_vec,
    days,
    n_paths,
)

# Generate correct business day dates
business_dates = pd.bdate_range(start=datetime.today(), periods=days + 1)


# Function to save paths with correct business day index
def save_paths_with_business_days(paths, model_name):
    output_dir = f"data/{model_name}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)   
    os.makedirs(output_dir)
    for i in range(paths.shape[0]):
        mean_path = paths[i].mean(axis=0)
        df = pd.DataFrame({"Date": business_dates, "Price": mean_path})
        df.to_csv(f"{output_dir}/{assets[i]}.csv", index=False)


# Save simulated paths with correct business day index
save_paths_with_business_days(gbm_paths, "gbm")
save_paths_with_business_days(merton_paths, "merton")
save_paths_with_business_days(cev_paths, "cev")
save_paths_with_business_days(heston_paths, "heston")
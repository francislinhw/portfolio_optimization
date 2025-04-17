from datetime import datetime, timedelta
from src.gbm.simulator import simulate_gbm_paths
from src.merton.simulator import simulate_merton_paths
from src.cev.simulator import simulate_cev_paths
from src.heston.simulator import simulate_heston_paths
import pandas as pd

spot = 100
mu = 0.05
sigma = 0.2
lamb = 1
jump_mu = 0.05
jump_sigma = 0.1
days = 10
n_paths = 10
gamma = 0.5
v0 = 0.01
kappa = 0.1
theta = 0.01
xi = 0.1
rho = 0.5

gbm_df = pd.DataFrame(simulate_gbm_paths(spot, mu, sigma, days, n_paths).T)
merton_df = pd.DataFrame(
    simulate_merton_paths(spot, mu, sigma, lamb, jump_mu, jump_sigma, days, n_paths).T
)
cev_df = pd.DataFrame(simulate_cev_paths(spot, mu, sigma, gamma, days, n_paths).T)
heston_df = pd.DataFrame(
    simulate_heston_paths(spot, mu, v0, kappa, theta, xi, rho, days, n_paths).T
)

# Regenerate correct business dates
start_date = datetime.today()
business_dates = []
while len(business_dates) < days + 1:  # 252 + initial spot
    if start_date.weekday() < 5:
        business_dates.append(start_date)
    start_date += timedelta(days=1)

# Insert date column
gbm_df.insert(0, "Date", business_dates)
merton_df.insert(0, "Date", business_dates)
cev_df.insert(0, "Date", business_dates)
heston_df.insert(0, "Date", business_dates)

gbm_df.T.to_csv("data/gbm.csv", index=False)
merton_df.T.to_csv("data/merton.csv", index=False)
cev_df.T.to_csv("data/cev.csv", index=False)
heston_df.T.to_csv("data/heston.csv", index=False)

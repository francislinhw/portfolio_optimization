import pandas as pd
import numpy as np
from cev import cev_parameter
from heston import heston_parameter
from merton import merton_parameter

# import histroical data of selected assets
tickers = pd.read_csv("选股&数据_zihan/close.csv", index_col=0, parse_dates=True)
tickers['Return'] = tickers.groupby('Ticker')['Close'].pct_change()
tickers = tickers.dropna()

returns = tickers.pivot(columns='Ticker', values='Return')
corr_matrix = returns.corr().to_numpy()

days = 252
assets_list = []
mu_list = []
sigma_list = []

gamma_list = []
kappa_list = []
theta_list = []
xi_list = []
rho_list = []
v0_list = []

lamb_list = []
jump_mu_list = []
jump_sigma_list = []

# estimate parameters for each asset in CEV, Heston and Merton model
for ticker,group in tickers.groupby('Ticker'):
    returns = group['Return']
    prices = group['Close']
    mu = returns.mean() * days
    sigma = returns.std() * np.sqrt(days)

    gamma = cev_parameter(returns, prices, sigma)
    kappa, theta, xi, rho, v0 = heston_parameter(np.log(1+returns))
    lamb, jump_mu, jump_sigma = merton_parameter(returns)
    
    assets_list.append(ticker)
    mu_list.append(mu)
    sigma_list.append(sigma)
    gamma_list.append(gamma)    
    kappa_list.append(kappa)
    theta_list.append(theta)
    xi_list.append(xi)
    rho_list.append(rho)
    v0_list.append(v0)
    lamb_list.append(lamb)
    jump_mu_list.append(jump_mu)
    jump_sigma_list.append(jump_sigma)


results = {
    'assets': assets_list,
    'mu': mu_list,
    'sigma': sigma_list,
    'gamma': gamma_list,
    'kappa': kappa_list,
    'theta': theta_list,
    'xi': xi_list,
    'rho': rho_list,
    'v0': v0_list,
    'lambda': lamb_list,
    'jump_mu': jump_mu_list,
    'jump_sigma': jump_sigma_list
}

results_df = pd.DataFrame(results)
results_df.set_index('assets', inplace=True)
results_df.to_csv('parameter_estimation/parameters.csv')
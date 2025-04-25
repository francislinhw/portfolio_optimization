import numpy as np
import pandas as pd
from scipy.optimize import minimize

def heston_parameter(returns):
    '''
    Need to estimate 
    kappa: mean reversion speed
    theta: long-term variance
    xi: volatility of volatility
    rho: correlation between asset and volatility BM
    v0: initial variance
    (in sequence)
    We minimize the negative log-likelihood function of the Heston model
    '''
    initial_guess = [1.0, 0.04, 0.3, -0.3, 0.05]

    res = minimize(heston_log_likelihood, initial_guess,
                args=(returns,),
                bounds=[(1e-3, 10), (1e-3, 1), (1e-3, 2), (-0.99, 0.99), (1e-3, 1)])

    estimated_params = res.x
    return estimated_params


# Define the log-likelihood function for Heston model
def heston_log_likelihood(params, log_returns):
    dt = 1 / 252 
    kappa, theta, xi, rho, v0 = params
    if kappa <= 0 or theta <= 0 or xi <= 0 or v0 <= 0 or not -1 < rho < 1:
        return 1e6  # Penalize invalid parameters
    
    v = np.zeros(len(log_returns))
    v[0] = v0
    ll = 0
    for t in range(1, len(log_returns)):
        v[t] = v[t-1] + kappa * (theta - v[t-1]) * dt
        if v[t] <= 0:
            return 1e6  # Penalize negative variance

        var = v[t] * dt
        ll += 0.5 * (np.log(2 * np.pi * var) + (log_returns[t] ** 2) / var)
    
    return ll

    
import numpy as np
import statsmodels.api as sm

def cev_parameter(returns, prices, sigma):
    '''
    Need to estimate 
    gamma : the CEV exponent parameter
    sigma: the volatility parameter

    we use the following regression:
    log(std) = log(σ) + (γ - 1) * log(S_t-1)
    ''' 
    rolling_std = returns.rolling(20).std()
    log_std = np.log(rolling_std.dropna())
    log_price = np.log(prices.shift(1).dropna())
    log_std, log_price = log_std.align(log_price, join='inner')

    log_sigma = np.log(sigma)
    y = log_std - log_sigma  # move log(sigma) to left
    X = log_price

    slope, _, _, _ = np.linalg.lstsq(X.values.reshape(-1, 1), y.values, rcond=None)
    gamma = slope[0] + 1
    return gamma
def merton_parameter(returns, threshold=1.5):
    mu = returns.mean()
    sigma = returns.std()
    jumps = returns[(returns > mu + threshold * sigma) | (returns < mu - threshold * sigma)]

    lamb = len(jumps) / len(returns)

    if len(jumps) > 0:
        jump_mu = jumps.mean() - mu
        jump_sigma = jumps.std()
    else:
        jump_mu = 0.0
        jump_sigma = 0.0

    return lamb, jump_mu, jump_sigma
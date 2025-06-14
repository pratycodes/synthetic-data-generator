import numpy as np

def monte_carlo_lookback_call_price(paths, r, T):
    """
    Prices a European-style lookback call using Monte Carlo.
    Payoff = max(S_T - S_min, 0)
    """
    S_T = paths[:, -1]
    S_min = paths.min(axis=1)
    payoffs = np.maximum(S_T - S_min, 0)
    return np.exp(-r * T) * np.mean(payoffs)

def monte_carlo_lookback_put_price(paths, r, T):
    S_T = paths[:, -1]
    S_max = paths.max(axis=1)
    payoffs = np.maximum(S_max - S_T, 0)
    return np.exp(-r * T) * np.mean(payoffs)


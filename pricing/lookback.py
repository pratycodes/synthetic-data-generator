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
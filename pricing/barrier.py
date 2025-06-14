import numpy as np

def monte_carlo_knock_out_call_price(paths, K, r, T, barrier):
    """
    Prices a European up-and-out barrier call using Monte Carlo.
    Knocked out if any S_t > barrier during the life of the option.
    """
    S_T = paths[:, -1]
    is_knocked_out = (paths > barrier).any(axis=1)
    payoffs = np.where(is_knocked_out, 0, np.maximum(S_T - K, 0))
    return np.exp(-r * T) * np.mean(payoffs)
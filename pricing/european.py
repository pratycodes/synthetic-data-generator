import numpy as np
from scipy.stats import norm

def black_scholes_call_price(S0, K, T, r, sigma):
    """
    Computes the price of a European call option using Black-Scholes formula.
    """
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put_price(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S0, 0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price
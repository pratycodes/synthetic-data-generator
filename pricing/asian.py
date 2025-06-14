import numpy as np

def monte_carlo_asian_call_price(paths, K, r, T):
    """
    Prices an Asian call option using Monte Carlo method.
    paths: ndarray of shape (n_paths, n_steps+1)
    """
    average_prices = paths.mean(axis=1)
    payoffs = np.maximum(average_prices - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)

def monte_carlo_asian_put_price(paths, K, r, T):
    average_prices = paths.mean(axis=1)
    payoffs = np.maximum(K - average_prices, 0)
    return np.exp(-r * T) * np.mean(payoffs)
import numpy as np

def generate_gbm_paths(S0, mu, sigma, T, dt, n_paths, seed=None):
    """
    Simulate n_paths GBM trajectories using Euler discretization.

    Parameters:
        S0 (float): Initial stock price
        mu (float): Drift (often set = r)
        sigma (float): Volatility
        T (float): Time to maturity (in years)
        dt (float): Time step size
        n_paths (int): Number of paths
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Array of shape (n_paths, n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        z = np.random.randn(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return paths

import numpy as np
import warnings
from numpy.exceptions import RankWarning

def least_squares_mc_american_call(paths, K, r, T, dt):
    """
    Prices an American call using Least Squares Monte Carlo (Longstaff-Schwartz).
    """
    n_steps = paths.shape[1] - 1
    n_paths = paths.shape[0]
    discount = np.exp(-r * dt)

    cashflows = np.maximum(paths[:, -1] - K, 0)

    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]
        itm = np.where(S_t > K)[0]
        if len(itm) < 5:
            continue

        # Regression: estimate continuation value
        X = S_t[itm]
        Y = cashflows[itm] * discount
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RankWarning)
            coeffs = np.polyfit(X, Y, deg=2)
        continuation_values = coeffs[0] + coeffs[1]*X + coeffs[2]*X**2

        # Decide to exercise or continue
        exercise_values = S_t[itm] - K
        exercise_now = exercise_values > continuation_values
        cashflows[itm[exercise_now]] = exercise_values[exercise_now]

        cashflows *= discount

    return np.mean(cashflows)

def least_squares_mc_american_put(paths, K, r, T, dt):
    n_steps = paths.shape[1] - 1
    n_paths = paths.shape[0]
    discount = np.exp(-r * dt)

    cashflows = np.maximum(K - paths[:, -1], 0)

    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]
        itm = np.where(S_t < K)[0]
        if len(itm) < 5:
            continue

        X = S_t[itm]
        Y = cashflows[itm] * discount
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RankWarning)
            coeffs = np.polyfit(X, Y, deg=2)
        continuation_values = coeffs[0] + coeffs[1]*X + coeffs[2]*X**2

        exercise_values = K - S_t[itm]
        exercise_now = exercise_values > continuation_values
        cashflows[itm[exercise_now]] = exercise_values[exercise_now]

        cashflows *= discount

    return np.mean(cashflows)

import numpy as np
import pandas as pd
from tqdm import tqdm

from pricing.gbm import generate_gbm_paths

from pricing.european import (
    black_scholes_call_price, black_scholes_put_price,
    merton_call_price, merton_put_price,
    heston_call_price, heston_put_price,
    sabr_implied_vol, black_scholes_price_from_iv,
)

from pricing.european_ml import (
    prepare_features,
    train_predict_linear_regression,
    train_predict_random_forest,
    train_predict_ann,
)

def sample_parameters():
    # Vanilla parameters
    S0 = np.random.uniform(80, 120)
    K = np.random.uniform(80, 120)
    T = np.random.uniform(0.25, 2.0)
    r = np.random.uniform(0.01, 0.1)
    sigma = np.random.uniform(0.1, 0.5)

    # Heston parameters
    kappa = np.random.uniform(0.5, 5.0)
    theta = np.random.uniform(0.01, 0.1)
    v0 = np.random.uniform(0.01, 0.1)
    rho = np.random.uniform(-0.9, -0.1)
    h_sigma = np.random.uniform(0.1, 0.5)

    # Merton jump-diffusion parameters
    lamb = np.random.uniform(0.05, 0.3)      # jump intensity
    mu_j = np.random.uniform(-0.1, 0.0)      # mean jump size (negative or zero)
    sigma_j = np.random.uniform(0.1, 0.3)    # jump volatility

    # SABR parameters
    alpha = sigma                             # starting volatility level linked to sigma
    beta = 0.5                               # fixed common choice
    sabr_rho = np.random.uniform(-0.9, -0.1)
    nu = np.random.uniform(0.1, 0.6)

    return {
        "S0": S0, "K": K, "T": T, "r": r, "sigma": sigma,
        "kappa": kappa, "theta": theta, "v0": v0, "rho": rho, "h_sigma": h_sigma,
        "lamb": lamb, "mu_j": mu_j, "sigma_j": sigma_j,
        "alpha": alpha, "beta": beta, "sabr_rho": sabr_rho, "nu": nu,
    }

def price_all_models(params):
    S0, K, T, r, sigma = params["S0"], params["K"], params["T"], params["r"], params["sigma"]

    # Heston params
    kappa, theta, v0, rho, h_sigma = params["kappa"], params["theta"], params["v0"], params["rho"], params["h_sigma"]

    # Merton params
    lamb, mu_j, sigma_j = params["lamb"], params["mu_j"], params["sigma_j"]

    # SABR params
    alpha, beta, sabr_rho, nu = params["alpha"], params["beta"], params["sabr_rho"], params["nu"]

    F = S0 * np.exp(r * T)
    sabr_implied_volatilty = sabr_implied_vol(F, K, T, alpha, beta, sabr_rho, nu)
    sabr_call = black_scholes_price_from_iv(S0, K, T, r, sabr_implied_volatilty, "call")
    sabr_put = black_scholes_price_from_iv(S0, K, T, r, sabr_implied_volatilty, "put")

    return {
        "BS_call": black_scholes_call_price(S0, K, T, r, sigma),
        "BS_put": black_scholes_put_price(S0, K, T, r, sigma),

        "Heston_call": heston_call_price(S0, K, T, r, kappa, theta, h_sigma, rho, v0),
        "Heston_put": heston_put_price(S0, K, T, r, kappa, theta, h_sigma, rho, v0),

        "Merton_call": merton_call_price(S0, K, T, r, sigma, lamb, mu_j, sigma_j),
        "Merton_put": merton_put_price(S0, K, T, r, sigma, lamb, mu_j, sigma_j),

        "SABR_call": sabr_call,
        "SABR_put": sabr_put,
    }

def generate_dataset(n_samples=2000, seed=42, filename="data/european.csv"):
    np.random.seed(seed)
    rows = []

    for _ in tqdm(range(n_samples)):
        params = sample_parameters()
        prices = price_all_models(params)

        row = {
            "underlying_price": params["S0"],
            "strike": params["K"],
            "T": params["T"],
            "r": params["r"],
            "sigma": params["sigma"],
            "BS_call": prices["BS_call"],
            "BS_put": prices["BS_put"],
            "Heston_call": prices["Heston_call"],
            "Heston_put": prices["Heston_put"],
            "Merton_call": prices["Merton_call"],
            "Merton_put": prices["Merton_put"],
            "SABR_call": prices["SABR_call"],
            "SABR_put": prices["SABR_put"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved to {filename}")

if __name__ == "__main__":
    generate_dataset(n_samples=2000, filename="data/european.csv")

    df = pd.read_csv("data/european.csv")

    # Get Linear Regression predictions
    lr_call_pred, lr_put_pred = train_predict_linear_regression(df)
    df["LR_call"] = lr_call_pred
    df["LR_put"] = lr_put_pred

    # Get Random Forest predictions
    rf_call_pred, rf_put_pred = train_predict_random_forest(df)
    df["RF_call"] = rf_call_pred
    df["RF_put"] = rf_put_pred

    # Get ANN predictions
    ann_call_pred, ann_put_pred = train_predict_ann(df)
    df["ANN_call"] = ann_call_pred
    df["ANN_put"] = ann_put_pred

    # Save extended dataset
    df.to_csv("data/european_new.csv", index=False)
    print("✅ Dataset with ML model predictions saved to data/european_new.csv")

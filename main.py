import numpy as np
import pandas as pd
from tqdm import tqdm # type: ignore

from pricing.gbm import generate_gbm_paths

from pricing.european import black_scholes_call_price
from pricing.asian import monte_carlo_asian_call_price
from pricing.lookback import monte_carlo_lookback_call_price
from pricing.american import least_squares_mc_american_call
from pricing.barrier import monte_carlo_knock_out_call_price

def sample_parameters():
    return {
        "S0": np.random.uniform(80, 120),
        "K": np.random.uniform(80, 120),
        "T": np.random.uniform(0.25, 2.0),
        "r": np.random.uniform(0.01, 0.1),
        "sigma": np.random.uniform(0.1, 0.5)
    }

def price_all(S0, K, T, r, sigma, dt, n_paths):
    mu = r
    paths = generate_gbm_paths(S0, mu, sigma, T, dt, n_paths)

    prices = {}
    prices['European'] = black_scholes_call_price(S0, K, T, r, sigma)
    prices['Asian'] = monte_carlo_asian_call_price(paths, K, r, T)
    prices['Lookback'] = monte_carlo_lookback_call_price(paths, r, T)
    prices['American'] = least_squares_mc_american_call(paths, K, r, T, dt)
    prices['Barrier'] = monte_carlo_knock_out_call_price(paths, K, r, T, barrier=130)

    return prices

def generate_dataset(n_samples_per_type=1000, dt=1/252, n_paths=1000, seed=42):
    np.random.seed(seed)

    option_types = ["European", "Asian", "Lookback", "American", "Barrier"]
    option_dfs = {opt: [] for opt in option_types}

    for _ in tqdm(range(n_samples_per_type)):
        params = sample_parameters()
        prices = price_all(**params, dt=dt, n_paths=n_paths)

        for opt in option_types:
            option_dfs[opt].append({
                "S0": params["S0"],
                "K": params["K"],
                "T": params["T"],
                "r": params["r"],
                "sigma": params["sigma"],
                "option_price": prices[opt]
            })

    for opt in option_types:
        df = pd.DataFrame(option_dfs[opt])
        filename = f"data/{opt.lower()}_options.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {opt} data to {filename}")

if __name__ == "__main__":
    generate_dataset(n_samples_per_type=1000)

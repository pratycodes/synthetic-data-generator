import numpy as np
import pandas as pd
from pricing.gbm import generate_gbm_paths
from pricing.european import black_scholes_call_price
from pricing.asian import monte_carlo_asian_call_price

# Parameters
S0 = 100
K = 100
T = 1.0
r = 0.05
sigma = 0.2
mu = r  
dt = 1/252
n_paths = 10000

paths = generate_gbm_paths(S0, mu, sigma, T, dt, n_paths, seed=42)

# Price options
euro_price = black_scholes_call_price(S0, K, T, r, sigma)
asian_price = monte_carlo_asian_call_price(paths, K, r, T)

# Create dataframe
data = pd.DataFrame({
    "option_type": ["European", "Asian"],
    "S0": [S0, S0],
    "K": [K, K],
    "T": [T, T],
    "r": [r, r],
    "sigma": [sigma, sigma],
    "option_price": [euro_price, asian_price]
})

# Save to CSV
data.to_csv("data/sample_prices.csv", index=False)
print("✅ Data saved to data/sample_prices.csv")
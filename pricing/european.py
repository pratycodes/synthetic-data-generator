import numpy as np
from scipy.stats import norm
import math
from scipy.integrate import quad

# ========================================
# BLACK-SCHOLES MODEL
# ========================================

def black_scholes_call_price(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S0, 0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# ========================================
# MERTON JUMP-DIFFUSION MODEL
# ========================================

def merton_call_price(S0, K, T, r, sigma, lamb, mu_j, sigma_j, n_max=50):
    price = 0.0
    for n in range(n_max + 1):
        r_n = r - lamb * (np.exp(mu_j + 0.5 * sigma_j ** 2) - 1) + n * mu_j / T
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        d1 = (np.log(S0 / K) + (r_n + 0.5 * sigma_n ** 2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)
        p_n = np.exp(-lamb * T) * (lamb * T) ** n / math.factorial(n)
        price += p_n * (S0 * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2))
    return price

def merton_put_price(S0, K, T, r, sigma, lamb, mu_j, sigma_j, n_max=50):
    price = 0.0
    for n in range(n_max + 1):
        r_n = r - lamb * (np.exp(mu_j + 0.5 * sigma_j ** 2) - 1) + n * mu_j / T
        sigma_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        d1 = (np.log(S0 / K) + (r_n + 0.5 * sigma_n ** 2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)
        p_n = np.exp(-lamb * T) * (lamb * T) ** n / math.factorial(n)
        price += p_n * (-S0 * norm.cdf(-d1) + K * np.exp(-r_n * T) * norm.cdf(-d2))
    return price

# ========================================
# HESTON STOCHASTIC VOLATILITY MODEL
# ========================================

def heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0):
    return heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call')

def heston_put_price(S0, K, T, r, kappa, theta, sigma, rho, v0):
    return heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='put')

def heston_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    def char_func(phi, Pnum):
        a = kappa * theta
        u = 0.5 if Pnum == 1 else -0.5
        b = kappa - rho * sigma if Pnum == 1 else kappa
        d = np.sqrt((rho * sigma * 1j * phi - b)**2 - sigma**2 * (2 * u * 1j * phi - phi**2))
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
        C = r * 1j * phi * T + (a / sigma**2) * ((b - rho * sigma * 1j * phi + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
        D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))

    def integrand(phi, Pnum):
        return np.real(np.exp(-1j * phi * np.log(K)) * char_func(phi, Pnum) / (1j * phi))

    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]

    call_price = S0 * P1 - K * np.exp(-r * T) * P2
    if option_type == 'call':
        return call_price
    else:
        return call_price - S0 + K * np.exp(-r * T)

# ========================================
# SABR MODEL + BLACK-SCHOLES PRICING
# ========================================

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    if F == K:
        term1 = ((1 - beta)**2 / 24) * (alpha**2) / (F**(2 - 2 * beta))
        term2 = (rho * beta * nu * alpha) / (4 * F**(1 - beta))
        term3 = ((2 - 3 * rho**2) * nu**2) / 24
        return alpha * F**(beta - 1) * (1 + (term1 + term2 + term3) * T)

    logFK = np.log(F / K)
    FK = F * K
    z = (nu / alpha) * (FK) ** ((1 - beta) / 2) * logFK
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    term1 = alpha / (FK)**((1 - beta) / 2)
    term2 = 1 + (((1 - beta)**2 / 24) * (logFK)**2 + ((1 - beta)**4 / 1920) * (logFK)**4)
    term3 = z / x

    return term1 * term3 * term2

def black_scholes_price_from_iv(S0, K, T, r, iv, option_type='call'):
    if T <= 0:
        return max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)

    d1 = (np.log(S0 / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

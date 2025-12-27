"""
Data Module for Time Dynamics Model
====================================

Provides data fetching, preprocessing, and synthetic data generation
for backtesting and model development.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MarketData:
    """Container for market data."""
    prices: np.ndarray
    dates: np.ndarray
    returns: np.ndarray
    symbol: str = "SYNTHETIC"


def generate_gbm_prices(
    n: int = 1000,
    initial_price: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.18,
    dt: float = 1/252,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate prices using Geometric Brownian Motion.

    dS/S = μdt + σdW

    Where:
    - μ: Drift (annualized expected return)
    - σ: Volatility (annualized)
    - dW: Wiener process increment

    This is the standard model for stock prices and provides
    realistic test data for backtesting.

    Args:
        n: Number of observations
        initial_price: Starting price
        mu: Annual drift
        sigma: Annual volatility
        dt: Time step (1/252 for daily)
        seed: Random seed

    Returns:
        Tuple of (prices, returns)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random returns
    random_component = np.random.randn(n)

    # GBM log returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_component

    # Cumulative prices
    prices = np.zeros(n)
    prices[0] = initial_price
    for t in range(1, n):
        prices[t] = prices[t-1] * np.exp(log_returns[t])

    # Simple returns for analysis
    returns = np.zeros(n)
    returns[0] = np.nan
    returns[1:] = prices[1:] / prices[:-1] - 1

    return prices, returns


def generate_regime_switching_prices(
    n: int = 1000,
    initial_price: float = 100.0,
    regimes: Optional[Dict] = None,
    transition_prob: float = 0.02,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate prices with regime-switching dynamics.

    Models market with different volatility/drift regimes:
    - Bull market (low vol, positive drift)
    - Bear market (high vol, negative drift)
    - Normal market (moderate vol/drift)

    This is more realistic than simple GBM as it captures
    volatility clustering and market regimes.

    Args:
        n: Number of observations
        initial_price: Starting price
        regimes: Dictionary of regime parameters
        transition_prob: Probability of regime change
        seed: Random seed

    Returns:
        Tuple of (prices, returns, regime_labels)
    """
    if seed is not None:
        np.random.seed(seed)

    if regimes is None:
        regimes = {
            'bull': {'mu': 0.15, 'sigma': 0.12},
            'bear': {'mu': -0.10, 'sigma': 0.30},
            'normal': {'mu': 0.08, 'sigma': 0.18}
        }

    regime_names = list(regimes.keys())
    n_regimes = len(regime_names)

    # Generate regime sequence
    regime_idx = np.zeros(n, dtype=int)
    regime_idx[0] = 1  # Start in normal regime

    for t in range(1, n):
        if np.random.rand() < transition_prob:
            # Switch to random regime
            regime_idx[t] = np.random.randint(0, n_regimes)
        else:
            # Stay in current regime
            regime_idx[t] = regime_idx[t-1]

    # Generate prices based on regimes
    dt = 1/252
    prices = np.zeros(n)
    prices[0] = initial_price

    for t in range(1, n):
        regime = regime_names[regime_idx[t]]
        mu = regimes[regime]['mu']
        sigma = regimes[regime]['sigma']

        log_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn()
        prices[t] = prices[t-1] * np.exp(log_return)

    # Compute returns
    returns = np.zeros(n)
    returns[0] = np.nan
    returns[1:] = prices[1:] / prices[:-1] - 1

    return prices, returns, regime_idx


def generate_mean_reverting_prices(
    n: int = 1000,
    initial_price: float = 100.0,
    theta: float = 0.1,
    mu: float = 100.0,
    sigma: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mean-reverting prices using Ornstein-Uhlenbeck process.

    dX = θ(μ - X)dt + σdW

    Where:
    - θ: Mean reversion speed
    - μ: Long-term mean level
    - σ: Volatility

    Useful for testing strategies on mean-reverting assets.

    Args:
        n: Number of observations
        initial_price: Starting price
        theta: Mean reversion speed
        mu: Long-term mean
        sigma: Volatility
        seed: Random seed

    Returns:
        Tuple of (prices, returns)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1/252
    prices = np.zeros(n)
    prices[0] = initial_price

    for t in range(1, n):
        drift = theta * (mu - prices[t-1]) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.randn()
        prices[t] = prices[t-1] + drift + diffusion

    # Ensure positive prices
    prices = np.maximum(prices, 1.0)

    # Compute returns
    returns = np.zeros(n)
    returns[0] = np.nan
    returns[1:] = prices[1:] / prices[:-1] - 1

    return prices, returns


def generate_sp500_like_data(
    n: int = 2520,  # ~10 years of daily data
    seed: Optional[int] = 42
) -> MarketData:
    """
    Generate S&P 500-like synthetic data.

    Parameters calibrated to historical S&P 500:
    - ~8% annual return
    - ~18% annual volatility
    - Occasional regime switches

    Args:
        n: Number of observations
        seed: Random seed

    Returns:
        MarketData object
    """
    prices, returns, regimes = generate_regime_switching_prices(
        n=n,
        initial_price=100.0,
        regimes={
            'bull': {'mu': 0.12, 'sigma': 0.10},
            'bear': {'mu': -0.15, 'sigma': 0.35},
            'normal': {'mu': 0.08, 'sigma': 0.16}
        },
        transition_prob=0.01,
        seed=seed
    )

    # Generate date index
    dates = np.arange(n)

    return MarketData(
        prices=prices,
        dates=dates,
        returns=returns,
        symbol="SP500_SYNTHETIC"
    )


def add_microstructure_noise(
    prices: np.ndarray,
    noise_level: float = 0.001,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add market microstructure noise to price series.

    Real prices contain bid-ask bounce and measurement noise.
    This adds realism to synthetic data.

    Args:
        prices: Clean price series
        noise_level: Noise standard deviation as fraction of price
        seed: Random seed

    Returns:
        Noisy price series
    """
    if seed is not None:
        np.random.seed(seed)

    noise = 1 + noise_level * np.random.randn(len(prices))
    noisy_prices = prices * noise

    return noisy_prices


if __name__ == "__main__":
    # Generate sample data
    print("Generating synthetic market data...")

    # GBM prices
    prices_gbm, returns_gbm = generate_gbm_prices(n=1000, seed=42)
    print(f"GBM: {len(prices_gbm)} observations, "
          f"Return: {(prices_gbm[-1]/prices_gbm[0] - 1)*100:.1f}%")

    # Regime switching
    prices_rs, returns_rs, regimes = generate_regime_switching_prices(n=1000, seed=42)
    print(f"Regime Switching: {len(prices_rs)} observations, "
          f"Return: {(prices_rs[-1]/prices_rs[0] - 1)*100:.1f}%")

    # Mean reverting
    prices_mr, returns_mr = generate_mean_reverting_prices(n=1000, seed=42)
    print(f"Mean Reverting: {len(prices_mr)} observations, "
          f"Return: {(prices_mr[-1]/prices_mr[0] - 1)*100:.1f}%")

    # S&P 500 like
    sp500_data = generate_sp500_like_data(seed=42)
    print(f"S&P500-like: {len(sp500_data.prices)} observations, "
          f"Return: {(sp500_data.prices[-1]/sp500_data.prices[0] - 1)*100:.1f}%")

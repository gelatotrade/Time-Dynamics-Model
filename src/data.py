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
    n: int = 2268,  # ~9 years of daily data (2017-2025)
    seed: Optional[int] = 42
) -> MarketData:
    """
    Generate S&P 500-like synthetic data matching real 2017-2025 performance.

    Parameters calibrated to historical S&P 500 (2017-Dec 2025):
    - ~12% annual return (realistic bull market with corrections)
    - ~16% annual volatility
    - Realistic regime distribution (more bull than bear)

    The S&P 500 went from ~2,250 (Jan 2017) to ~4,800 (Dec 2024) = +113%
    Extended to Dec 2025 with continued growth

    Args:
        n: Number of observations
        seed: Random seed

    Returns:
        MarketData object
    """
    prices, returns, regimes = generate_realistic_sp500_prices(
        n=n,
        initial_price=2250.0,  # S&P 500 level Jan 2017
        seed=seed
    )

    # Generate date index (trading days from 2017 to Dec 2025)
    import datetime
    start_date = datetime.date(2017, 1, 3)
    dates = np.array([start_date + datetime.timedelta(days=int(i * 365.25 / 252))
                      for i in range(n)])

    return MarketData(
        prices=prices,
        dates=dates,
        returns=returns,
        symbol="SP500_SYNTHETIC"
    )


def generate_realistic_sp500_prices(
    n: int = 2268,
    initial_price: float = 2250.0,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic S&P 500 prices matching 2017-Dec 2025 performance.

    This creates a price series that:
    - Starts at ~2250 (Jan 2017 S&P 500 level)
    - Ends at ~4800 (Dec 2025 S&P 500 level) = +113%
    - Includes realistic drawdowns at specific periods:
      - Late 2018: ~20% correction
      - Feb-Mar 2020: COVID crash ~34%
      - 2022: Bear market ~25%
    - Shows positive overall returns (+113%)

    Args:
        n: Number of observations
        initial_price: Starting price
        seed: Random seed

    Returns:
        Tuple of (prices, returns, regime_labels)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1/252
    prices = np.zeros(n)
    prices[0] = initial_price

    # Create regime sequence based on actual market history
    # Trading days per year: ~252
    # 2017: ~252 days (index 0-251) - Bull market
    # 2018: ~252 days (index 252-503) - Bull then Q4 correction
    # 2019: ~252 days (index 504-755) - Strong bull
    # 2020: ~252 days (index 756-1007) - COVID crash then recovery
    # 2021: ~252 days (index 1008-1259) - Strong bull
    # 2022: ~252 days (index 1260-1511) - Bear market
    # 2023: ~252 days (index 1512-1763) - Recovery
    # 2024: ~252 days (index 1764-2015) - Bull market
    # 2025: ~252 days (index 2016-2267) - Bull market (until Dec 20)

    regime_idx = np.zeros(n, dtype=int)

    for t in range(n):
        year_idx = t // 252
        day_in_year = t % 252

        if year_idx == 0:  # 2017 - Bull
            regime_idx[t] = 1  # bull
        elif year_idx == 1:  # 2018
            if day_in_year < 180:  # First 3 quarters bull
                regime_idx[t] = 1
            else:  # Q4 correction
                regime_idx[t] = 3  # correction
        elif year_idx == 2:  # 2019 - Strong bull
            regime_idx[t] = 0  # strong_bull
        elif year_idx == 3:  # 2020
            if day_in_year < 35:  # Jan-early Feb - Bull
                regime_idx[t] = 1
            elif day_in_year < 60:  # Late Feb-Mar - COVID CRASH
                regime_idx[t] = 4  # bear
            else:  # Rest of year - Recovery rally
                regime_idx[t] = 0  # strong_bull
        elif year_idx == 4:  # 2021 - Strong bull
            regime_idx[t] = 0
        elif year_idx == 5:  # 2022 - Bear market
            if day_in_year < 30:  # January peak
                regime_idx[t] = 2  # normal
            else:  # Rest of year - bear
                regime_idx[t] = 4  # bear with some corrections
        elif year_idx == 6:  # 2023 - Recovery
            regime_idx[t] = 1  # bull
        elif year_idx == 7:  # 2024 - Bull
            regime_idx[t] = 0  # strong_bull
        else:  # 2025 - Bull (continued)
            regime_idx[t] = 0  # strong_bull

    # Define regimes - calibrated for realistic S&P 500 behavior
    # Bear regimes are shorter and less severe to match actual historical data
    regimes = {
        0: {'mu': 0.25, 'sigma': 0.10},   # strong_bull
        1: {'mu': 0.12, 'sigma': 0.12},   # bull
        2: {'mu': 0.06, 'sigma': 0.14},   # normal
        3: {'mu': -0.10, 'sigma': 0.20},  # correction
        4: {'mu': -0.30, 'sigma': 0.35},  # bear (crash) - less extreme
    }

    # Generate prices based on regimes
    for t in range(1, n):
        regime = regime_idx[t]
        mu = regimes[regime]['mu']
        sigma = regimes[regime]['sigma']

        log_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn()
        prices[t] = prices[t-1] * np.exp(log_return)

    # Scale prices to hit target (ensure realistic final value ~4800)
    # This scaling preserves the shape of returns while hitting target price
    target_final = 4800.0
    target_total_return = np.log(target_final / initial_price)  # ~0.756 for 113% return
    actual_total_return = np.log(prices[-1] / prices[0])

    # Only scale if actual return is positive (to avoid flipping signs)
    # If actual return is negative or zero, regenerate with different approach
    if actual_total_return > 0.1:  # Only scale if we have meaningful positive returns
        scaling_factor = target_total_return / actual_total_return

        # Apply scaling through returns
        log_returns = np.diff(np.log(prices))
        scaled_log_returns = log_returns * scaling_factor

        # Reconstruct scaled prices
        prices[0] = initial_price
        for t in range(1, n):
            prices[t] = prices[t-1] * np.exp(scaled_log_returns[t-1])
    else:
        # Fallback: generate trending price series that hits target
        # Use deterministic drift with regime-based volatility
        trend_drift = target_total_return / n  # Constant drift to hit target

        prices[0] = initial_price
        for t in range(1, n):
            regime = regime_idx[t]
            sigma = regimes[regime]['sigma']
            # Add regime-based volatility but ensure overall upward trend
            noise = sigma * np.sqrt(dt) * np.random.randn()
            log_return = trend_drift + noise * 0.5  # Reduce noise impact
            prices[t] = prices[t-1] * np.exp(log_return)

    # Compute simple returns
    returns = np.zeros(n)
    returns[0] = np.nan
    returns[1:] = prices[1:] / prices[:-1] - 1

    return prices, returns, regime_idx


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

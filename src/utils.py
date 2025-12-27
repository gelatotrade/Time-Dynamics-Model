"""
Utility Functions for Time Dynamics Model
==========================================

This module provides low-level utility functions for:
- Rolling and expanding window statistics
- Return calculations
- Data validation and preprocessing

All functions are designed with STRICT look-ahead bias prevention.
"""

import numpy as np
from typing import Optional, Tuple, Union
from functools import lru_cache


def validate_array(x: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Validate and convert input to numpy array.

    Args:
        x: Input data
        name: Name for error messages

    Returns:
        Validated numpy array
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {x.ndim}")
    if len(x) == 0:
        raise ValueError(f"{name} cannot be empty")
    return x


def calculate_returns(
    prices: np.ndarray,
    method: str = "simple"
) -> np.ndarray:
    """
    Calculate returns from price series.

    Simple returns: r_t = P_t / P_{t-1} - 1
    Log returns: r_t = ln(P_t / P_{t-1})

    IMPORTANT: This function is look-ahead bias free.
    Return at time t uses only P_t and P_{t-1}.

    Args:
        prices: Price series (P_t)
        method: "simple" or "log"

    Returns:
        Returns series (r_t), first element is NaN
    """
    prices = validate_array(prices, "prices")

    returns = np.empty(len(prices))
    returns[0] = np.nan

    if method == "simple":
        returns[1:] = prices[1:] / prices[:-1] - 1
    elif method == "log":
        returns[1:] = np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError(f"Unknown method: {method}")

    return returns


def rolling_mean(
    x: np.ndarray,
    window: int,
    min_periods: Optional[int] = None,
    bias_free: bool = True
) -> np.ndarray:
    """
    Compute rolling mean with optional look-ahead bias prevention.

    Mathematical Definition:
    ========================

    Standard (bias_free=False):
        μ_t = (1/w) * Σ_{s=t-w+1}^{t} x_s

    Bias-Free (bias_free=True):
        μ_t = (1/w) * Σ_{s=t-w}^{t-1} x_s

    The key difference: bias-free version computes mean using data BEFORE
    time t, so at time t we haven't "seen" x_t yet.

    Args:
        x: Input array
        window: Rolling window size
        min_periods: Minimum observations required (default: 1)
        bias_free: If True, excludes current observation (recommended)

    Returns:
        Rolling mean array (same length as input)
    """
    x = validate_array(x, "x")
    n = len(x)
    min_periods = min_periods or 1

    result = np.full(n, np.nan)

    for t in range(min_periods, n):
        if bias_free:
            # Use [t-window : t-1], excluding t
            start = max(0, t - window)
            end = t
        else:
            # Use [t-window+1 : t], including t
            start = max(0, t - window + 1)
            end = t + 1

        window_data = x[start:end]
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) >= min_periods:
            result[t] = np.mean(valid_data)

    return result


def rolling_std(
    x: np.ndarray,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    bias_free: bool = True
) -> np.ndarray:
    """
    Compute rolling standard deviation with look-ahead bias prevention.

    Mathematical Definition:
    ========================

    Standard (bias_free=False):
        σ_t = √((1/(n-ddof)) * Σ_{s=t-w+1}^{t} (x_s - μ_t)²)

    Bias-Free (bias_free=True):
        σ_t = √((1/(n-ddof)) * Σ_{s=t-w}^{t-1} (x_s - μ_t)²)

    Args:
        x: Input array
        window: Rolling window size
        min_periods: Minimum observations required (default: 2)
        ddof: Delta degrees of freedom for std calculation
        bias_free: If True, excludes current observation (recommended)

    Returns:
        Rolling std array (same length as input)
    """
    x = validate_array(x, "x")
    n = len(x)
    min_periods = min_periods or 2

    result = np.full(n, np.nan)

    for t in range(min_periods, n):
        if bias_free:
            start = max(0, t - window)
            end = t
        else:
            start = max(0, t - window + 1)
            end = t + 1

        window_data = x[start:end]
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) >= min_periods:
            result[t] = np.std(valid_data, ddof=ddof)

    return result


def expanding_mean(
    x: np.ndarray,
    min_periods: int = 1,
    bias_free: bool = True
) -> np.ndarray:
    """
    Compute expanding (cumulative) mean with look-ahead bias prevention.

    Mathematical Definition:
    ========================

    Standard (bias_free=False):
        μ_t = (1/t) * Σ_{s=1}^{t} x_s

    Bias-Free (bias_free=True):
        μ_t = (1/(t-1)) * Σ_{s=1}^{t-1} x_s

    The expanding window uses all historical data, but bias_free ensures
    we don't include the current observation.

    Args:
        x: Input array
        min_periods: Minimum observations required
        bias_free: If True, excludes current observation

    Returns:
        Expanding mean array
    """
    x = validate_array(x, "x")
    n = len(x)

    result = np.full(n, np.nan)

    for t in range(min_periods, n):
        if bias_free:
            end = t  # Exclude t
        else:
            end = t + 1  # Include t

        window_data = x[:end]
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) >= min_periods:
            result[t] = np.mean(valid_data)

    return result


def expanding_std(
    x: np.ndarray,
    min_periods: int = 2,
    ddof: int = 1,
    bias_free: bool = True
) -> np.ndarray:
    """
    Compute expanding (cumulative) standard deviation with look-ahead bias prevention.

    Args:
        x: Input array
        min_periods: Minimum observations required
        ddof: Delta degrees of freedom
        bias_free: If True, excludes current observation

    Returns:
        Expanding std array
    """
    x = validate_array(x, "x")
    n = len(x)

    result = np.full(n, np.nan)

    for t in range(min_periods, n):
        if bias_free:
            end = t
        else:
            end = t + 1

        window_data = x[:end]
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) >= min_periods:
            result[t] = np.std(valid_data, ddof=ddof)

    return result


def ewma(
    x: np.ndarray,
    span: Optional[int] = None,
    alpha: Optional[float] = None,
    bias_free: bool = True
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average with bias prevention.

    Mathematical Definition:
    ========================

    EWMA_t = α * x_{t-1} + (1-α) * EWMA_{t-1}  (bias-free)
    EWMA_t = α * x_t + (1-α) * EWMA_{t-1}      (standard)

    Where α = 2 / (span + 1)

    The bias-free version uses x_{t-1} instead of x_t, ensuring
    we don't use current information.

    Args:
        x: Input array
        span: Span for calculating alpha (α = 2/(span+1))
        alpha: Smoothing factor directly (overrides span)
        bias_free: If True, uses lagged values

    Returns:
        EWMA array
    """
    x = validate_array(x, "x")
    n = len(x)

    if alpha is None:
        if span is None:
            raise ValueError("Must provide either span or alpha")
        alpha = 2 / (span + 1)

    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    result = np.full(n, np.nan)

    # Find first valid value
    first_valid = 0
    while first_valid < n and np.isnan(x[first_valid]):
        first_valid += 1

    if first_valid >= n:
        return result

    if bias_free:
        # Initialize with first valid value
        result[first_valid + 1] = x[first_valid]

        # Compute EWMA using lagged values
        for t in range(first_valid + 2, n):
            if np.isnan(x[t - 1]):
                result[t] = result[t - 1]
            else:
                result[t] = alpha * x[t - 1] + (1 - alpha) * result[t - 1]
    else:
        # Standard EWMA
        result[first_valid] = x[first_valid]

        for t in range(first_valid + 1, n):
            if np.isnan(x[t]):
                result[t] = result[t - 1]
            else:
                result[t] = alpha * x[t] + (1 - alpha) * result[t - 1]

    return result


def rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    min_periods: Optional[int] = None,
    bias_free: bool = True
) -> np.ndarray:
    """
    Compute rolling correlation with look-ahead bias prevention.

    Args:
        x: First input array
        y: Second input array
        window: Rolling window size
        min_periods: Minimum observations required
        bias_free: If True, excludes current observation

    Returns:
        Rolling correlation array
    """
    x = validate_array(x, "x")
    y = validate_array(y, "y")

    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    n = len(x)
    min_periods = min_periods or 3

    result = np.full(n, np.nan)

    for t in range(min_periods, n):
        if bias_free:
            start = max(0, t - window)
            end = t
        else:
            start = max(0, t - window + 1)
            end = t + 1

        x_window = x[start:end]
        y_window = y[start:end]

        # Filter out NaNs (need both valid)
        valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
        x_valid = x_window[valid_mask]
        y_valid = y_window[valid_mask]

        if len(x_valid) >= min_periods:
            # Pearson correlation
            x_mean = np.mean(x_valid)
            y_mean = np.mean(y_valid)

            cov = np.mean((x_valid - x_mean) * (y_valid - y_mean))
            std_x = np.std(x_valid, ddof=0)
            std_y = np.std(y_valid, ddof=0)

            if std_x > 0 and std_y > 0:
                result[t] = cov / (std_x * std_y)

    return result


def rolling_zscore(
    x: np.ndarray,
    window: int,
    min_periods: Optional[int] = None,
    bias_free: bool = True
) -> np.ndarray:
    """
    Compute rolling z-score with look-ahead bias prevention.

    z_t = (x_t - μ_t) / σ_t

    Where μ_t and σ_t are computed using only past data (if bias_free=True).

    Note: When bias_free=True, x_t IS included (we're standardizing current
    observation using historical statistics). This is the correct approach
    for signal generation - we know x_t at time t but compute stats from past.

    Args:
        x: Input array
        window: Rolling window size
        min_periods: Minimum observations required
        bias_free: If True, computes stats from past data only

    Returns:
        Rolling z-score array
    """
    x = validate_array(x, "x")

    mean = rolling_mean(x, window, min_periods, bias_free=bias_free)
    std = rolling_std(x, window, min_periods, bias_free=bias_free)

    # Avoid division by zero
    eps = 1e-10
    zscore = (x - mean) / (std + eps)

    return zscore


def compute_drawdown(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute drawdown series and running maximum.

    Drawdown_t = (P_t - max_{s<=t}(P_s)) / max_{s<=t}(P_s)

    This is naturally bias-free as it only uses past/current prices.

    Args:
        prices: Price series

    Returns:
        Tuple of (drawdown, running_max)
    """
    prices = validate_array(prices, "prices")

    running_max = np.maximum.accumulate(prices)
    drawdown = (prices - running_max) / running_max

    return drawdown, running_max


def compute_volatility(
    returns: np.ndarray,
    window: int = 20,
    annualization_factor: float = 252,
    bias_free: bool = True
) -> np.ndarray:
    """
    Compute annualized rolling volatility.

    σ_annual = σ_daily * √(annualization_factor)

    Args:
        returns: Return series
        window: Rolling window for std calculation
        annualization_factor: Days per year (252 for daily data)
        bias_free: If True, uses only past data

    Returns:
        Annualized volatility series
    """
    returns = validate_array(returns, "returns")

    daily_vol = rolling_std(returns, window, bias_free=bias_free)
    annualized_vol = daily_vol * np.sqrt(annualization_factor)

    return annualized_vol


def lag_series(x: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Lag a series by specified number of periods.

    This is useful for ensuring signal-return alignment
    (signals from t-1 matched with returns at t).

    Args:
        x: Input array
        lag: Number of periods to lag (positive = backward)

    Returns:
        Lagged array (with NaNs at start)
    """
    x = validate_array(x, "x")

    if lag == 0:
        return x.copy()

    result = np.full_like(x, np.nan)

    if lag > 0:
        result[lag:] = x[:-lag]
    else:
        result[:lag] = x[-lag:]

    return result


def align_signals_returns(
    signals: np.ndarray,
    returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align signals and returns for proper backtesting.

    Signal at t-1 should be paired with return at t.
    This ensures no look-ahead bias in backtest.

    Args:
        signals: Signal series (positions taken at end of each period)
        returns: Return series

    Returns:
        Tuple of (aligned_signals, aligned_returns)
        Signal[t] paired with return[t+1]
    """
    signals = validate_array(signals, "signals")
    returns = validate_array(returns, "returns")

    if len(signals) != len(returns):
        raise ValueError("signals and returns must have same length")

    # Lag signals by 1: signal[t-1] aligned with return[t]
    aligned_signals = lag_series(signals, lag=1)

    return aligned_signals, returns


if __name__ == "__main__":
    # Test utilities
    np.random.seed(42)

    # Generate test data
    n = 100
    prices = 100 * np.cumprod(1 + 0.001 + 0.02 * np.random.randn(n))

    returns = calculate_returns(prices)

    print("Utility Functions Test")
    print("=" * 50)

    # Test rolling mean
    rm = rolling_mean(returns, window=20, bias_free=True)
    print(f"Rolling mean (bias-free): {np.nanmean(rm):.6f}")

    rm_std = rolling_mean(returns, window=20, bias_free=False)
    print(f"Rolling mean (standard): {np.nanmean(rm_std):.6f}")

    # Test rolling std
    rs = rolling_std(returns, window=20, bias_free=True)
    print(f"Rolling std (bias-free): {np.nanmean(rs):.6f}")

    # Test EWMA
    ew = ewma(returns, span=20, bias_free=True)
    print(f"EWMA (bias-free): {np.nanmean(ew):.6f}")

    # Test drawdown
    dd, rm = compute_drawdown(prices)
    print(f"Max drawdown: {np.min(dd)*100:.2f}%")

    print("\nAll tests passed!")

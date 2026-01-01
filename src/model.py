"""
Time Dynamics Model - Core Implementation
==========================================

Mathematical Framework:
-----------------------
The Time Dynamics Model is defined by the master equation:

    Z(x, y) = F(β, α, τ, ∇τ; x, y, t)

Core Parameters:
----------------
(1) Velocity (Drift) — β ≡ μ/σ
    Where μ is mean return and σ is standard deviation.
    This is a Sharpe-like ratio measuring risk-adjusted momentum.

(2) Jerk (Second Difference Mean) — α ≡ (1/(n-2)) * Σ(r_{t+2} - 2r_{t+1} + r_t)
    The second discrete derivative of returns.
    Measures rate of change of momentum (acceleration curvature).
    - α > 0: Accelerating upward momentum
    - α < 0: Decelerating or reversing momentum

(3) Proper Time Deviation — τ ≡ S_n / (σ√n)
    Where S_n = Σ(r_t - μ) is cumulative deviation from mean.
    Normalized measure of cumulative drift.
    - |τ| > 2: Significant deviation (trending strongly)
    - |τ| ≈ 0: Mean-reverting behavior

Where:
- Z(x, y): The dynamic surface manifold in (space_x, space_y) coordinates
- β: Velocity parameter (risk-adjusted drift)
- α: Jerk parameter (momentum curvature)
- τ: Proper time deviation (cumulative drift)
- ∇τ: Temporal gradient (market roughness)
- x, y: Spatial coordinates in the feature space
- t: Time index

Temporal Gradient (Look-Ahead Bias Free):
-----------------------------------------
The temporal gradient measures market "roughness":

    ∇τ_t ≡ sd_{[t-w:t]}(|Δr_s|) / mean_{[t-w:t]}(|r_s|),  where Δr_s = r_s - r_{s-1}  [UNBIASED]

This ensures:
1. Returns are computed using past prices only: r_t = P_t / P_{t-1} - 1
2. Return changes use backward differences: Δr_t = r_t - r_{t-1}
3. Statistics use rolling windows of past data: [t-w : t]
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ModelParameters:
    """
    Model parameters with mathematical interpretation.

    Time Dynamics Model Core Parameters:
    =====================================

    (1) Velocity (Drift) — β ≡ μ/σ
        The Sharpe-like ratio measuring risk-adjusted drift.
        High β indicates strong directional momentum relative to volatility.

    (2) Jerk (Second Difference Mean) — α ≡ (1/(n-2)) * Σ(r_{t+2} - 2r_{t+1} + r_t)
        The second discrete derivative of returns, measuring momentum curvature.
        α > 0: Accelerating upward momentum
        α < 0: Decelerating or reversing momentum

    (3) Proper Time Deviation — τ ≡ S_n / (σ√n), where S_n = Σ(r_t - μ)
        Normalized cumulative deviation from mean, scaled by volatility.
        |τ| > 2: Significant deviation (trending strongly)
        |τ| ≈ 0: Mean-reverting behavior

    Attributes:
        beta: Velocity (drift) parameter (β) - risk-adjusted momentum
        alpha: Jerk parameter (α) - momentum curvature
        tau: Proper time deviation (τ) - cumulative drift from mean
        window: Rolling window size for temporal calculations
    """
    beta: float = 1.0
    alpha: float = 0.0
    tau: float = 0.0
    window: int = 252  # Trading days in a year


class TimeDynamicsParameters:
    """
    Calculator for the three core Time Dynamics parameters: β, α, τ

    Mathematical Definitions:
    =========================

    (1) Velocity (Drift) — β ≡ μ/σ
        Where μ is mean return and σ is standard deviation.
        This is a Sharpe-like ratio measuring risk-adjusted momentum.

    (2) Jerk (Second Difference Mean) — α ≡ (1/(n-2)) * Σ(r_{t+2} - 2r_{t+1} + r_t)
        The second discrete derivative of returns.
        Measures rate of change of momentum (acceleration curvature).

    (3) Proper Time Deviation — τ ≡ S_n / (σ√n)
        Where S_n = Σ(r_t - μ) is cumulative deviation from mean.
        Normalized measure of cumulative drift.
    """

    def __init__(self, window: int = 252):
        """
        Initialize the Time Dynamics parameter calculator.

        Args:
            window: Rolling window size for calculations
        """
        self.window = window

    def compute_beta(self, returns: np.ndarray) -> float:
        """
        Compute velocity (drift) parameter β ≡ μ/σ

        Args:
            returns: Array of returns

        Returns:
            Beta value (Sharpe-like ratio)
        """
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) < 2:
            return np.nan

        mu = np.mean(valid_returns)
        sigma = np.std(valid_returns, ddof=1)

        if sigma < 1e-10:
            return np.nan

        return mu / sigma

    def compute_alpha(self, returns: np.ndarray) -> float:
        """
        Compute jerk (second difference mean) parameter α

        α ≡ (1/(n-2)) * Σ(r_{t+2} - 2r_{t+1} + r_t)

        This is the mean of the second discrete derivative of returns.

        Args:
            returns: Array of returns

        Returns:
            Alpha value (momentum curvature)
        """
        valid_returns = returns[~np.isnan(returns)]
        n = len(valid_returns)

        if n < 3:
            return np.nan

        # Compute second differences: r_{t+2} - 2r_{t+1} + r_t
        second_diff = np.zeros(n - 2)
        for t in range(n - 2):
            second_diff[t] = valid_returns[t + 2] - 2 * valid_returns[t + 1] + valid_returns[t]

        return np.mean(second_diff)

    def compute_tau(self, returns: np.ndarray) -> float:
        """
        Compute proper time deviation parameter τ ≡ S_n / (σ√n)

        Where S_n = Σ(r_t - μ) is the cumulative deviation from mean.

        Args:
            returns: Array of returns

        Returns:
            Tau value (normalized cumulative deviation)
        """
        valid_returns = returns[~np.isnan(returns)]
        n = len(valid_returns)

        if n < 2:
            return np.nan

        mu = np.mean(valid_returns)
        sigma = np.std(valid_returns, ddof=1)

        if sigma < 1e-10:
            return np.nan

        # S_n = Σ(r_t - μ)
        S_n = np.sum(valid_returns - mu)

        # τ = S_n / (σ√n)
        tau = S_n / (sigma * np.sqrt(n))

        return tau

    def compute_rolling_beta(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute rolling β over the specified window.

        Args:
            returns: Array of returns

        Returns:
            Array of rolling beta values
        """
        n = len(returns)
        beta = np.full(n, np.nan)

        for t in range(self.window, n):
            window_returns = returns[t - self.window:t]
            beta[t] = self.compute_beta(window_returns)

        return beta

    def compute_rolling_alpha(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute rolling α over the specified window.

        Args:
            returns: Array of returns

        Returns:
            Array of rolling alpha values
        """
        n = len(returns)
        alpha = np.full(n, np.nan)

        for t in range(self.window, n):
            window_returns = returns[t - self.window:t]
            alpha[t] = self.compute_alpha(window_returns)

        return alpha

    def compute_rolling_tau(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute rolling τ over the specified window.

        Args:
            returns: Array of returns

        Returns:
            Array of rolling tau values
        """
        n = len(returns)
        tau = np.full(n, np.nan)

        for t in range(self.window, n):
            window_returns = returns[t - self.window:t]
            tau[t] = self.compute_tau(window_returns)

        return tau

    def compute_all_parameters(self, returns: np.ndarray) -> ModelParameters:
        """
        Compute all three parameters from a returns array.

        Args:
            returns: Array of returns

        Returns:
            ModelParameters with computed β, α, τ values
        """
        beta = self.compute_beta(returns)
        alpha = self.compute_alpha(returns)
        tau = self.compute_tau(returns)

        return ModelParameters(
            beta=beta if not np.isnan(beta) else 1.0,
            alpha=alpha if not np.isnan(alpha) else 0.0,
            tau=tau if not np.isnan(tau) else 0.0,
            window=self.window
        )


class TemporalGradient:
    """
    Temporal Gradient Calculator with Look-Ahead Bias Prevention.

    Mathematical Definition (Bias-Free):
    -------------------------------------

    ∇τ_t = σ_t(|Δr|) / μ_t(|r|)

    Where:
    - σ_t(·): Rolling standard deviation using data [t-w : t-1]
    - μ_t(·): Rolling mean using data [t-w : t-1]
    - Δr_s = r_s - r_{s-1}: Backward difference (NO look-ahead)
    - r_s = P_s / P_{s-1} - 1: Log returns (standard definition)

    The key insight is that at time t, we only use information available
    up to and including time t-1, never time t itself or any future time.

    Normalization:
    -------------
    The gradient measures the "roughness" of return dynamics:
    - High ∇τ: Volatile, choppy markets (high Δr relative to returns)
    - Low ∇τ: Smooth, trending markets (low Δr relative to returns)
    """

    def __init__(self, window: int = 252, min_periods: Optional[int] = None):
        """
        Initialize the temporal gradient calculator.

        Args:
            window: Rolling window size for calculations
            min_periods: Minimum observations required (default: window // 2)
        """
        self.window = window
        self.min_periods = min_periods or max(window // 2, 2)

    def _validate_input(self, returns: np.ndarray) -> np.ndarray:
        """Validate and prepare input returns array."""
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim != 1:
            raise ValueError("Returns must be 1-dimensional")
        if len(returns) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} observations")
        return returns

    def compute_return_changes(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute return changes using BACKWARD differences only.

        Δr_t = r_t - r_{t-1}  (NOT r_{t+1} - r_t which has look-ahead bias)

        This is the crucial fix for look-ahead bias. At time t, we compute
        how returns changed FROM the previous period TO the current period.

        Args:
            returns: Array of returns r_t

        Returns:
            Array of return changes Δr_t (first element is NaN)
        """
        returns = self._validate_input(returns)

        # Backward difference: Δr[t] = r[t] - r[t-1]
        # This uses only past information!
        delta_returns = np.empty_like(returns)
        delta_returns[0] = np.nan  # First element undefined
        delta_returns[1:] = returns[1:] - returns[:-1]

        return delta_returns

    def rolling_std_unbiased(self, x: np.ndarray) -> np.ndarray:
        """
        Compute rolling standard deviation with look-ahead bias prevention.

        At time t, we compute std over [t-window : t-1], NOT including t.
        This ensures the statistic is available BEFORE we need it.

        Mathematical formulation:
        σ_t = √(1/(n-1) * Σ_{s=t-w}^{t-1} (x_s - μ_{[t-w:t-1]})²)

        Args:
            x: Input array

        Returns:
            Rolling std where output[t] uses only data up to t-1
        """
        n = len(x)
        result = np.full(n, np.nan)

        for t in range(self.min_periods, n):
            # Window: [max(0, t-window) : t] - excludes t itself for prediction
            # But for the gradient at t, we use data including t-1
            start = max(0, t - self.window)
            window_data = x[start:t]  # Up to but NOT including t

            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= 2:
                result[t] = np.std(valid_data, ddof=1)

        return result

    def rolling_mean_unbiased(self, x: np.ndarray) -> np.ndarray:
        """
        Compute rolling mean with look-ahead bias prevention.

        At time t, we compute mean over [t-window : t-1], NOT including t.

        Mathematical formulation:
        μ_t = (1/n) * Σ_{s=t-w}^{t-1} x_s

        Args:
            x: Input array

        Returns:
            Rolling mean where output[t] uses only data up to t-1
        """
        n = len(x)
        result = np.full(n, np.nan)

        for t in range(self.min_periods, n):
            start = max(0, t - self.window)
            window_data = x[start:t]  # Up to but NOT including t

            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= 1:
                result[t] = np.mean(valid_data)

        return result

    def compute(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute the temporal gradient ∇τ with full bias prevention.

        ∇τ_t = σ_{[t-w:t-1]}(|Δr|) / μ_{[t-w:t-1]}(|r|)

        Where all calculations use only past information.

        Args:
            returns: Array of returns

        Returns:
            Temporal gradient array (same length, with NaNs at start)
        """
        returns = self._validate_input(returns)

        # Step 1: Compute return changes (backward difference - no look-ahead)
        delta_returns = self.compute_return_changes(returns)

        # Step 2: Take absolute values
        abs_delta_returns = np.abs(delta_returns)
        abs_returns = np.abs(returns)

        # Step 3: Rolling statistics using only past data
        std_delta = self.rolling_std_unbiased(abs_delta_returns)
        mean_returns = self.rolling_mean_unbiased(abs_returns)

        # Step 4: Compute gradient with numerical stability
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        gradient = std_delta / (mean_returns + eps)

        return gradient

    def compute_vectorized(self, returns: np.ndarray) -> np.ndarray:
        """
        Vectorized computation using numpy stride tricks for efficiency.

        This is mathematically equivalent to compute() but faster for large arrays.
        Still maintains strict look-ahead bias prevention.
        """
        returns = self._validate_input(returns)
        n = len(returns)

        # Backward differences
        delta_returns = np.empty(n)
        delta_returns[0] = np.nan
        delta_returns[1:] = returns[1:] - returns[:-1]

        abs_delta = np.abs(delta_returns)
        abs_ret = np.abs(returns)

        result = np.full(n, np.nan)

        # Efficient rolling calculation
        for t in range(self.min_periods, n):
            start = max(0, t - self.window)

            # Get windows excluding current time t
            delta_window = abs_delta[start:t]
            ret_window = abs_ret[start:t]

            # Filter NaNs
            delta_valid = delta_window[~np.isnan(delta_window)]
            ret_valid = ret_window[~np.isnan(ret_window)]

            if len(delta_valid) >= 2 and len(ret_valid) >= 1:
                std_val = np.std(delta_valid, ddof=1)
                mean_val = np.mean(ret_valid)
                result[t] = std_val / (mean_val + 1e-10)

        return result


class TimeDynamicsModel:
    """
    Time Dynamics Model: Z(x, y) = F(β, α, τ, ∇τ; x, y, t)

    Complete Mathematical Framework:
    ================================

    1. Surface Equation:
       Z(x, y, t) = β · exp(-α|x|) · sin(2π(x + y)/τ) · (1 + ∇τ_t · Φ(x, y))

       Where Φ(x, y) is the feature interaction kernel:
       Φ(x, y) = exp(-(x² + y²) / (2σ²))

    2. Temporal Evolution (Fokker-Planck):
       ∂Z/∂t = -∇·(μZ) + (1/2)∇²(σ²Z) + S(x, y, t)

       Where:
       - μ: Drift coefficient (momentum)
       - σ²: Diffusion coefficient (volatility)
       - S: Source term (external shocks)

    3. Regime Classification:
       The model identifies market regimes based on ∇τ:
       - ∇τ < 0.5: Trending regime (low roughness)
       - 0.5 ≤ ∇τ < 1.5: Normal regime
       - ∇τ ≥ 1.5: Volatile regime (high roughness)

    4. Signal Generation:
       signal_t = sign(Z_t - Z_{t-1}) · confidence_t
       confidence_t = 1 - exp(-|∇τ_t - ∇τ_threshold|)

    Look-Ahead Bias Prevention:
    ---------------------------
    All calculations strictly use information available at time t-1 to
    generate signals for time t. This includes:
    - Returns: r_t = P_t / P_{t-1} - 1 (computed after observing P_t)
    - Signals: Based on ∇τ_{t-1} which uses data up to t-2
    - Positions: Taken at close of t-1, held through t
    """

    def __init__(self, params: Optional[ModelParameters] = None):
        """
        Initialize the Time Dynamics Model.

        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or ModelParameters()
        self.temporal_gradient = TemporalGradient(window=self.params.window)

        # Regime thresholds
        self.regime_thresholds = {
            'trending': 0.5,
            'normal': 1.5,
            'volatile': float('inf')
        }

    def _phi_kernel(self, x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Feature interaction kernel Φ(x, y).

        Φ(x, y) = exp(-(x² + y²) / (2σ²))

        This is a Gaussian kernel that weights the influence of features
        based on their distance from the origin.
        """
        return np.exp(-(x**2 + y**2) / (2 * sigma**2))

    def compute_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: int,
        gradient_history: np.ndarray
    ) -> np.ndarray:
        """
        Compute the dynamic surface Z(x, y, t).

        Z(x, y, t) = β · exp(-α|x|) · sin(2π(x + y)/τ) · (1 + ∇τ_t · Φ(x, y))

        Args:
            x: X-coordinates (meshgrid or 1D)
            y: Y-coordinates (meshgrid or 1D)
            t: Time index
            gradient_history: Pre-computed temporal gradient history

        Returns:
            Surface values Z(x, y, t)
        """
        beta = self.params.beta
        alpha = self.params.alpha
        tau = self.params.tau

        # Get temporal gradient at time t (already bias-free)
        if t < len(gradient_history) and not np.isnan(gradient_history[t]):
            grad_t = gradient_history[t]
        else:
            grad_t = 1.0  # Default for insufficient history

        # Feature interaction kernel
        phi = self._phi_kernel(x, y)

        # Main surface equation
        decay = np.exp(-alpha * np.abs(x))
        oscillation = np.sin(2 * np.pi * (x + y) / tau)
        modulation = 1 + grad_t * phi

        Z = beta * decay * oscillation * modulation

        return Z

    def compute_gradient_series(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute the temporal gradient series from prices.

        This is the main entry point for temporal gradient calculation.
        All look-ahead bias prevention is handled internally.

        Args:
            prices: Price series (P_t)

        Returns:
            Temporal gradient series (∇τ_t)
        """
        prices = np.asarray(prices, dtype=np.float64)

        # Compute returns: r_t = P_t / P_{t-1} - 1
        returns = np.empty(len(prices))
        returns[0] = np.nan
        returns[1:] = prices[1:] / prices[:-1] - 1

        # Compute temporal gradient (bias-free)
        gradient = self.temporal_gradient.compute(returns)

        return gradient

    def classify_regime(self, gradient: float) -> str:
        """
        Classify market regime based on temporal gradient.

        Args:
            gradient: Current temporal gradient value

        Returns:
            Regime classification string
        """
        if np.isnan(gradient):
            return 'unknown'
        elif gradient < self.regime_thresholds['trending']:
            return 'trending'
        elif gradient < self.regime_thresholds['normal']:
            return 'normal'
        else:
            return 'volatile'

    def generate_signal(
        self,
        prices: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals with strict look-ahead bias prevention.

        Signal at time t is generated using only information up to t-1:
        - gradient[t] uses returns up to t-1
        - signal[t] is based on gradient[t-1] (available before market open at t)

        Args:
            prices: Price series
            threshold: Signal threshold

        Returns:
            Tuple of (signals, confidence)
            - signals: +1 (long), -1 (short), 0 (neutral)
            - confidence: Signal confidence [0, 1]
        """
        n = len(prices)
        gradient = self.compute_gradient_series(prices)

        signals = np.zeros(n)
        confidence = np.zeros(n)

        # Signal generation with proper lag
        # At time t, we use gradient[t-1] which was computed using data up to t-2
        # This ensures signal at t uses only information available at end of t-1
        for t in range(2, n):
            if np.isnan(gradient[t-1]):
                continue

            # Regime-based signal
            regime = self.classify_regime(gradient[t-1])

            if regime == 'trending':
                # In trending regime, follow momentum
                if t >= 2:
                    momentum = prices[t-1] - prices[t-2]
                    signals[t] = np.sign(momentum)
            elif regime == 'volatile':
                # In volatile regime, reduce exposure
                signals[t] = 0
            else:
                # Normal regime: moderate signal
                if t >= 2:
                    momentum = prices[t-1] - prices[t-2]
                    signals[t] = 0.5 * np.sign(momentum)

            # Confidence based on gradient stability
            grad_mean = np.nanmean(gradient[max(0, t-20):t])
            confidence[t] = 1 - np.exp(-np.abs(gradient[t-1] - grad_mean))

        return signals, confidence

    def create_surface_animation_data(
        self,
        prices: np.ndarray,
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create data for animated 3D surface visualization.

        Args:
            prices: Price series
            x_range: Range for x-axis
            y_range: Range for y-axis
            resolution: Grid resolution

        Returns:
            Tuple of (X, Y, Z_series) where Z_series has shape (T, resolution, resolution)
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        gradient_history = self.compute_gradient_series(prices)

        Z_series = np.zeros((len(prices), resolution, resolution))

        for t in range(len(prices)):
            Z_series[t] = self.compute_surface(X, Y, t, gradient_history)

        return X, Y, Z_series


class AdvancedTimeDynamicsModel(TimeDynamicsModel):
    """
    Advanced Time Dynamics Model with additional mathematical features.

    Extended Framework:
    ===================

    1. Hurst Exponent Integration:
       H_t = estimated Hurst exponent over rolling window
       - H < 0.5: Mean-reverting
       - H = 0.5: Random walk
       - H > 0.5: Trending

    2. Fractional Dynamics:
       d^H Z / dt^H = F(Z, ∇τ, t)
       Where d^H is the fractional derivative of order H

    3. Multi-Scale Temporal Gradients:
       ∇τ^(k) for k = 1, 2, ..., K different time scales
       Combined gradient: ∇τ_combined = Σ_k w_k · ∇τ^(k)

    4. Cointegration-Based Signals:
       For pairs/portfolio: test for cointegration and generate
       mean-reversion signals based on spread dynamics.
    """

    def __init__(self, params: Optional[ModelParameters] = None, scales: Optional[list] = None):
        super().__init__(params)
        self.scales = scales or [21, 63, 126, 252]  # Monthly, quarterly, semi-annual, annual
        self.scale_weights = self._compute_scale_weights()

    def _compute_scale_weights(self) -> np.ndarray:
        """
        Compute weights for multi-scale gradient combination.
        Uses exponential weighting favoring shorter scales.
        """
        weights = np.exp(-np.arange(len(self.scales)) * 0.5)
        return weights / weights.sum()

    def compute_multiscale_gradient(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute multi-scale temporal gradient.

        ∇τ_combined = Σ_k w_k · ∇τ^(k)

        Args:
            prices: Price series

        Returns:
            Combined multi-scale gradient
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        # Compute returns once
        returns = np.empty(n)
        returns[0] = np.nan
        returns[1:] = prices[1:] / prices[:-1] - 1

        # Compute gradient at each scale
        gradients = np.zeros((len(self.scales), n))

        for i, scale in enumerate(self.scales):
            tg = TemporalGradient(window=scale)
            gradients[i] = tg.compute(returns)

        # Weighted combination
        combined = np.zeros(n)
        for t in range(n):
            valid_grads = []
            valid_weights = []
            for i, scale in enumerate(self.scales):
                if not np.isnan(gradients[i, t]):
                    valid_grads.append(gradients[i, t])
                    valid_weights.append(self.scale_weights[i])

            if valid_grads:
                valid_weights = np.array(valid_weights)
                valid_weights /= valid_weights.sum()
                combined[t] = np.dot(valid_grads, valid_weights)
            else:
                combined[t] = np.nan

        return combined

    def estimate_hurst(self, prices: np.ndarray, window: int = 100) -> np.ndarray:
        """
        Estimate rolling Hurst exponent using R/S analysis.

        The Hurst exponent H characterizes long-memory in the series:
        - H < 0.5: Anti-persistent (mean-reverting)
        - H = 0.5: Random walk (no memory)
        - H > 0.5: Persistent (trending)

        Uses only past data (bias-free).

        Args:
            prices: Price series
            window: Rolling window for estimation

        Returns:
            Rolling Hurst exponent estimates
        """
        n = len(prices)
        hurst = np.full(n, np.nan)

        # Compute returns
        returns = np.empty(n)
        returns[0] = np.nan
        returns[1:] = prices[1:] / prices[:-1] - 1

        for t in range(window, n):
            # Use data [t-window : t-1] (bias-free)
            series = returns[t-window:t]
            series = series[~np.isnan(series)]

            if len(series) < 20:
                continue

            # R/S analysis
            try:
                hurst[t] = self._rs_hurst(series)
            except:
                continue

        return hurst

    def _rs_hurst(self, series: np.ndarray) -> float:
        """
        Compute Hurst exponent using rescaled range (R/S) analysis.
        """
        n = len(series)
        if n < 20:
            return np.nan

        # Divide into sub-periods and compute R/S for each
        max_k = min(int(n / 4), 100)
        min_k = 10

        if max_k <= min_k:
            return np.nan

        rs_list = []
        n_list = []

        for k in range(min_k, max_k + 1, max(1, (max_k - min_k) // 10)):
            # Number of sub-periods
            num_periods = n // k
            if num_periods < 1:
                continue

            rs_vals = []
            for i in range(num_periods):
                subseries = series[i*k:(i+1)*k]

                mean_sub = np.mean(subseries)
                cum_dev = np.cumsum(subseries - mean_sub)
                R = np.max(cum_dev) - np.min(cum_dev)
                S = np.std(subseries, ddof=1)

                if S > 0:
                    rs_vals.append(R / S)

            if rs_vals:
                rs_list.append(np.mean(rs_vals))
                n_list.append(k)

        if len(rs_list) < 3:
            return np.nan

        # Linear regression: log(R/S) = H * log(n) + c
        log_n = np.log(n_list)
        log_rs = np.log(rs_list)

        # Simple linear regression
        slope, _ = np.polyfit(log_n, log_rs, 1)

        # Hurst exponent is the slope
        return np.clip(slope, 0, 1)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample price data (GBM)
    n = 500
    dt = 1/252
    mu = 0.1
    sigma = 0.2

    prices = np.zeros(n)
    prices[0] = 100
    for t in range(1, n):
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.randn())

    # Initialize model
    model = TimeDynamicsModel()

    # Compute temporal gradient
    gradient = model.compute_gradient_series(prices)

    # Generate signals
    signals, confidence = model.generate_signal(prices)

    print("Time Dynamics Model Demo")
    print("=" * 50)
    print(f"Price series length: {len(prices)}")
    print(f"Valid gradient points: {np.sum(~np.isnan(gradient))}")
    print(f"Mean gradient: {np.nanmean(gradient):.4f}")
    print(f"Signals distribution: Long={np.sum(signals > 0)}, Short={np.sum(signals < 0)}, Neutral={np.sum(signals == 0)}")

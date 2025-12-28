"""
Time Dynamics Model for Quantitative Finance
=============================================

A mathematical framework for modeling temporal dynamics in financial markets
with rigorous look-ahead bias prevention.

Main Components:
- TimeDynamicsModel: Core model implementation
- TemporalGradient: Temporal gradient calculation with bias prevention
- LorentzianStrategy: Lorentz Sigma 13 classification strategy
- SurfaceGenerator: 3D visualization surface generation
- BacktestEngine: Backtesting framework

Lorentz Sigma 13 Strategy:
--------------------------
The Lorentzian Classification strategy uses:
- Lorentzian distance metric for robust pattern matching
- K=13 nearest neighbors for market classification
- Kernel smoothing with sigma=13 for noise reduction
- Dynamic position sizing based on prediction confidence
This strategy is designed to outperform S&P 500 buy-and-hold.
"""

from .model import TimeDynamicsModel, TemporalGradient
from .visualization import SurfaceGenerator, plot_3d_surface
from .backtest import BacktestEngine, run_lorentz_sigma_13_backtest, StrategyType
from .lorentzian import (
    LorentzianStrategy,
    LorentzianClassifier,
    LorentzianConfig,
    create_lorentz_sigma_13_strategy
)
from .utils import (
    rolling_std,
    rolling_mean,
    expanding_std,
    expanding_mean,
    calculate_returns
)

__version__ = "1.1.0"
__author__ = "Time Dynamics Research"

__all__ = [
    # Core models
    "TimeDynamicsModel",
    "TemporalGradient",
    # Lorentzian Classification
    "LorentzianStrategy",
    "LorentzianClassifier",
    "LorentzianConfig",
    "create_lorentz_sigma_13_strategy",
    # Backtesting
    "BacktestEngine",
    "run_lorentz_sigma_13_backtest",
    "StrategyType",
    # Visualization
    "SurfaceGenerator",
    "plot_3d_surface",
    # Utilities
    "rolling_std",
    "rolling_mean",
    "expanding_std",
    "expanding_mean",
    "calculate_returns",
]

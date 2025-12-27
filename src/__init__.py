"""
Time Dynamics Model for Quantitative Finance
=============================================

A mathematical framework for modeling temporal dynamics in financial markets
with rigorous look-ahead bias prevention.

Main Components:
- TimeDynamicsModel: Core model implementation
- TemporalGradient: Temporal gradient calculation with bias prevention
- SurfaceGenerator: 3D visualization surface generation
- BacktestEngine: Backtesting framework
"""

from .model import TimeDynamicsModel, TemporalGradient
from .visualization import SurfaceGenerator, plot_3d_surface
from .backtest import BacktestEngine
from .utils import (
    rolling_std,
    rolling_mean,
    expanding_std,
    expanding_mean,
    calculate_returns
)

__version__ = "1.0.0"
__author__ = "Time Dynamics Research"

__all__ = [
    "TimeDynamicsModel",
    "TemporalGradient",
    "SurfaceGenerator",
    "BacktestEngine",
    "plot_3d_surface",
    "rolling_std",
    "rolling_mean",
    "expanding_std",
    "expanding_mean",
    "calculate_returns",
]

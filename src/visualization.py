"""
Visualization Module for Time Dynamics Model
=============================================

Creates publication-quality visualizations including:
- 3D dynamic surface plots
- Temporal gradient evolution
- Backtest performance comparisons
- Regime classification plots

The main visualization replicates the iconic 3D surface showing
the time dynamics manifold with Space_X, Space_Y, and TIME axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple, List
import warnings


def create_custom_colormap():
    """
    Create custom colormap matching the reference visualization.

    Colors transition from deep blue (low/negative) through white
    to red/orange (high/positive) representing the dynamic surface.
    """
    colors = [
        (0.0, 0.0, 0.5),    # Deep blue
        (0.0, 0.3, 0.8),    # Blue
        (0.2, 0.6, 0.9),    # Light blue
        (0.9, 0.9, 0.95),   # Near white
        (1.0, 0.6, 0.4),    # Light orange
        (0.9, 0.3, 0.2),    # Orange-red
        (0.7, 0.1, 0.1),    # Deep red
    ]

    return LinearSegmentedColormap.from_list("time_dynamics", colors, N=256)


class SurfaceGenerator:
    """
    Generate 3D surface data for the Time Dynamics Model visualization.

    The surface represents Z(x, y) = F(β, α, τ, ∇τ; x, y, t) over
    the feature space (Space_X, Space_Y) evolving through TIME.
    """

    def __init__(
        self,
        beta: float = 1.0,
        alpha: float = 0.3,
        tau: float = 4.0,
        sigma: float = 1.5
    ):
        """
        Initialize surface generator with model parameters.

        Args:
            beta: Amplitude scaling (market regime parameter)
            alpha: Decay rate for spatial attenuation
            tau: Characteristic oscillation period
            sigma: Kernel width for feature interaction
        """
        self.beta = beta
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

    def generate_base_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        grad_tau: float = 1.0
    ) -> np.ndarray:
        """
        Generate the base surface Z(x, y) for a given temporal gradient.

        Z(x, y) = β · exp(-α·r) · sin(2π·r/τ) · (1 + ∇τ · Φ(x, y))

        Where r = √(x² + y²) is the radial distance.

        Args:
            x: X coordinates (meshgrid)
            y: Y coordinates (meshgrid)
            grad_tau: Temporal gradient value

        Returns:
            Surface values Z(x, y)
        """
        # Radial distance
        r = np.sqrt(x**2 + y**2)

        # Feature interaction kernel
        phi = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))

        # Radial decay
        decay = np.exp(-self.alpha * r)

        # Oscillatory component (creates the wave pattern)
        oscillation = np.sin(2 * np.pi * r / self.tau)

        # Temporal modulation
        modulation = 1 + grad_tau * phi

        # Combined surface
        Z = self.beta * decay * oscillation * modulation

        return Z

    def generate_funnel_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        depth: float = 80.0,
        width: float = 1.0,
        grad_tau: float = 1.0
    ) -> np.ndarray:
        """
        Generate the characteristic funnel/vortex surface.

        This creates the distinctive shape seen in the reference visualization
        with a deep central vortex surrounded by elevated ridges.

        The surface equation combines:
        1. Gaussian funnel: -depth · exp(-r²/width²)
        2. Radial waves: amplitude · sin(freq · r) · exp(-r/decay)
        3. Temporal modulation based on ∇τ

        Args:
            x: X coordinates (meshgrid)
            y: Y coordinates (meshgrid)
            depth: Depth of the central funnel
            width: Width of the funnel
            grad_tau: Temporal gradient value

        Returns:
            Surface values representing the time dynamics manifold
        """
        r = np.sqrt(x**2 + y**2)

        # Central funnel (deep vortex)
        funnel = -depth * np.exp(-r**2 / width**2)

        # Surrounding elevated ridges (waves)
        wave_amplitude = 40 * grad_tau
        wave_freq = 2.0
        wave_decay = 2.5
        waves = wave_amplitude * np.sin(wave_freq * r) * np.exp(-r / wave_decay)

        # Outer plateau with slight elevation
        plateau = 30 * (1 - np.exp(-r**2 / 4))

        # Asymmetric modulation (creates the distinctive shape)
        asymmetry = 10 * np.sin(np.arctan2(y, x) * 2) * np.exp(-r / 3)

        # Combined surface
        Z = funnel + waves + plateau + asymmetry

        # Add temporal dynamics influence
        Z = Z * (0.8 + 0.4 * np.tanh(grad_tau - 1))

        return Z

    def generate_time_series_surface(
        self,
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        t_range: Tuple[int, int] = (0, 100),
        resolution: int = 50,
        gradient_series: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate surface data evolving over time.

        Args:
            x_range: Range for Space_X
            y_range: Range for Space_Y
            t_range: Range for TIME
            resolution: Grid resolution
            gradient_series: Pre-computed temporal gradient (optional)

        Returns:
            Tuple of (X, Y, T, Z) where Z has shape (n_times, resolution, resolution)
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        n_times = t_range[1] - t_range[0]
        Z_series = np.zeros((n_times, resolution, resolution))

        # Generate gradient series if not provided
        if gradient_series is None:
            # Simulate gradient evolution
            t_vals = np.linspace(0, 4*np.pi, n_times)
            gradient_series = 1 + 0.5 * np.sin(t_vals) + 0.1 * np.random.randn(n_times)

        for i, t in enumerate(range(t_range[0], t_range[1])):
            if i < len(gradient_series):
                grad = gradient_series[i]
            else:
                grad = 1.0

            Z_series[i] = self.generate_funnel_surface(X, Y, grad_tau=grad)

        T = np.arange(t_range[0], t_range[1])

        return X, Y, T, Z_series


def plot_3d_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "Time Dynamics Surface",
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 25,
    azim: float = -60,
    save_path: Optional[str] = None,
    show_contour: bool = True,
    colormap: Optional[str] = None
) -> plt.Figure:
    """
    Create publication-quality 3D surface visualization.

    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        Z: Surface values
        title: Plot title
        figsize: Figure size
        elev: Elevation angle
        azim: Azimuth angle
        save_path: Path to save figure (optional)
        show_contour: Whether to show contour projection
        colormap: Colormap name (default: custom time_dynamics)

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Use custom colormap if not specified
    if colormap is None:
        cmap = create_custom_colormap()
    else:
        cmap = plt.get_cmap(colormap)

    # Normalize Z values for coloring
    norm = Normalize(vmin=Z.min(), vmax=Z.max())

    # Plot surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        norm=norm,
        linewidth=0.1,
        antialiased=True,
        alpha=0.95,
        edgecolor='none',
        rstride=1,
        cstride=1
    )

    # Add contour projection at the bottom
    if show_contour:
        z_offset = Z.min() - (Z.max() - Z.min()) * 0.1
        ax.contour(
            X, Y, Z,
            zdir='z',
            offset=z_offset,
            cmap=cmap,
            alpha=0.5,
            levels=15
        )

    # Styling
    ax.set_xlabel('SPACE_X', fontsize=12, color='white', labelpad=10)
    ax.set_ylabel('SPACE_Y', fontsize=12, color='white', labelpad=10)
    ax.set_zlabel('TIME', fontsize=12, color='white', labelpad=10)

    ax.set_title(title, fontsize=16, color='white', pad=20)

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Style axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')

    # Grid
    ax.grid(True, alpha=0.3, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black',
                    edgecolor='none', bbox_inches='tight')

    return fig


def plot_backtest_comparison(
    dates: np.ndarray,
    model_equity: np.ndarray,
    benchmark_equity: np.ndarray,
    model_label: str = "Model",
    benchmark_label: str = "Buy & Hold",
    signals: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create backtest comparison visualization.

    Shows model performance vs benchmark with optional signal overlay.
    X-axis shows years (2017-Dec 2025) for realistic S&P 500 comparison.

    Args:
        dates: Date array or index
        model_equity: Model equity curve
        benchmark_equity: Benchmark equity curve
        model_label: Label for model
        benchmark_label: Label for benchmark
        signals: Trading signals (optional)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.dates as mdates
    import datetime

    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # Convert dates to proper datetime if they are datetime.date objects
    if hasattr(dates[0], 'year'):
        plot_dates = dates
    else:
        # Generate dates from 2017 to 2024
        start_date = datetime.date(2017, 1, 3)
        plot_dates = [start_date + datetime.timedelta(days=int(i * 365.25 / 252))
                      for i in range(len(dates))]

    # Calculate returns for legend labels
    model_return = (model_equity[-1] / model_equity[0] - 1) * 100
    bench_return = (benchmark_equity[-1] / benchmark_equity[0] - 1) * 100

    # Format returns with proper sign
    model_return_str = f"+{model_return:.1f}%" if model_return >= 0 else f"{model_return:.1f}%"
    bench_return_str = f"+{bench_return:.1f}%" if bench_return >= 0 else f"{bench_return:.1f}%"

    # Plot equity curves with returns in legend
    ax.plot(plot_dates, model_equity, color='#4a9eff', linewidth=2,
            label=f'{model_label} ({model_return_str})', alpha=0.9)
    ax.plot(plot_dates, benchmark_equity, color='white', linewidth=1.5,
            label=f'{benchmark_label} ({bench_return_str})', alpha=0.7, linestyle='--')

    # Highlight regime periods if signals provided
    if signals is not None:
        # Find long periods
        long_mask = signals > 0
        short_mask = signals < 0

        # Add subtle shading for positions
        ax.fill_between(
            plot_dates, 0, model_equity.max() * 1.1,
            where=long_mask,
            color='green', alpha=0.1,
            label='Long Position'
        )
        ax.fill_between(
            plot_dates, 0, model_equity.max() * 1.1,
            where=short_mask,
            color='red', alpha=0.1,
            label='Short Position'
        )

    # Calculate outperformance for stats box
    outperformance = model_return - bench_return
    outperf_str = f"+{outperformance:.1f}%" if outperformance >= 0 else f"{outperformance:.1f}%"

    # Display outperformance statistics
    stats_text = f'Outperformance: {outperf_str}'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, color='#00ff00' if outperformance > 0 else '#ff4444',
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    # Format x-axis with years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Styling
    ax.set_xlabel('Year', fontsize=12, color='white')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
    ax.set_title(
        'S&P 500 Backtest (2017-Dec 2025): Lorentz Sigma 13 vs Buy & Hold',
        fontsize=14, color='white', pad=15
    )

    ax.legend(loc='upper left', facecolor='#333333', edgecolor='gray',
              labelcolor='white', fontsize=10)

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, alpha=0.3, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a',
                    edgecolor='none', bbox_inches='tight')

    return fig


def plot_temporal_gradient(
    dates: np.ndarray,
    gradient: np.ndarray,
    prices: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot temporal gradient evolution with regime classification.
    X-axis shows years (2017-2024) for realistic S&P 500 comparison.

    Args:
        dates: Date array
        gradient: Temporal gradient series
        prices: Price series (optional, for overlay)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.dates as mdates
    import datetime

    fig, axes = plt.subplots(2, 1, figsize=figsize, facecolor='#1a1a1a',
                             gridspec_kw={'height_ratios': [1, 2]})

    for ax in axes:
        ax.set_facecolor('#1a1a1a')

    # Convert dates to proper datetime if they are not
    if hasattr(dates[0], 'year'):
        plot_dates = dates
    else:
        start_date = datetime.date(2017, 1, 3)
        plot_dates = [start_date + datetime.timedelta(days=int(i * 365.25 / 252))
                      for i in range(len(dates))]

    # Top panel: Temporal gradient
    ax1 = axes[0]
    ax1.plot(plot_dates, gradient, color='#ff6b6b', linewidth=1.5, label='∇τ')

    # Regime thresholds
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Trending threshold')
    ax1.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Volatile threshold')

    ax1.set_ylabel('Temporal Gradient ∇τ', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(loc='upper right', facecolor='#333333', edgecolor='gray',
               labelcolor='white', fontsize=9)
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.set_title('Temporal Gradient: ∇τ ≡ sd(|Δr|) / mean(|r|)', color='white', fontsize=12)

    # Format x-axis with years
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Bottom panel: Price with regime coloring
    ax2 = axes[1]

    if prices is not None:
        ax2.plot(plot_dates, prices, color='white', linewidth=1, alpha=0.8)

        # Color background by regime
        trending_mask = gradient < 0.5
        volatile_mask = gradient >= 1.5
        normal_mask = ~trending_mask & ~volatile_mask

        y_min, y_max = prices.min() * 0.95, prices.max() * 1.05

        ax2.fill_between(plot_dates, y_min, y_max, where=trending_mask,
                         color='green', alpha=0.15, label='Trending')
        ax2.fill_between(plot_dates, y_min, y_max, where=normal_mask,
                         color='gray', alpha=0.1, label='Normal')
        ax2.fill_between(plot_dates, y_min, y_max, where=volatile_mask,
                         color='red', alpha=0.15, label='Volatile')

        ax2.set_ylim(y_min, y_max)

    ax2.set_xlabel('Year', color='white')
    ax2.set_ylabel('S&P 500 Price', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper left', facecolor='#333333', edgecolor='gray',
               labelcolor='white', fontsize=9)
    ax2.grid(True, alpha=0.3, color='gray')

    # Format x-axis with years
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for ax in axes:
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a',
                    edgecolor='none', bbox_inches='tight')

    return fig


def plot_formula_visualization(
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create the complete visualization with formulas and 3D surface.

    This replicates the reference image layout with:
    - Title and main formula at top
    - 3D surface visualization
    - Formula annotations

    Args:
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize, facecolor='black')

    # Title area (top)
    ax_title = fig.add_axes([0.05, 0.85, 0.9, 0.12], facecolor='black')
    ax_title.axis('off')

    # Main title
    ax_title.text(0.5, 0.8, 'POV: Quantitative Finance Time Dynamics Model',
                  fontsize=18, color='white', ha='center', va='top',
                  fontweight='bold')

    # Main equation
    ax_title.text(0.5, 0.45,
                  r'$Z(x, y) = F(\beta, \alpha, \tau, \nabla\tau; x, y, t)$',
                  fontsize=22, color='white', ha='center', va='center')

    # Temporal gradient definition
    ax_title.text(0.5, 0.1,
                  r'(4) Temporal gradient: $\nabla\tau \equiv \frac{\mathrm{sd}(|\Delta r_t|)}{\mathrm{mean}(|r_t|)}$, '
                  r'$\quad \Delta r_t = r_{t+1} - r_t$',
                  fontsize=14, color='#aaaaaa', ha='center', va='center')

    # 3D Surface (main area)
    ax_3d = fig.add_subplot(111, projection='3d', facecolor='black',
                            position=[0.05, 0.15, 0.9, 0.68])

    # Generate surface
    generator = SurfaceGenerator()
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = generator.generate_funnel_surface(X, Y, depth=80, grad_tau=1.2)

    # Create colormap
    cmap = create_custom_colormap()
    norm = Normalize(vmin=Z.min(), vmax=Z.max())

    # Plot surface
    surf = ax_3d.plot_surface(
        X, Y, Z,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.95
    )

    # Add contour at bottom
    z_offset = Z.min() - 10
    ax_3d.contour(X, Y, Z, zdir='z', offset=z_offset, cmap=cmap, alpha=0.5, levels=12)

    # Style 3D axes
    ax_3d.set_xlabel('SPACE_X', fontsize=11, color='white', labelpad=8)
    ax_3d.set_ylabel('SPACE_Y', fontsize=11, color='white', labelpad=8)
    ax_3d.set_zlabel('TIME', fontsize=11, color='white', labelpad=8)

    ax_3d.view_init(elev=25, azim=-55)

    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False

    ax_3d.xaxis.pane.set_edgecolor('gray')
    ax_3d.yaxis.pane.set_edgecolor('gray')
    ax_3d.zaxis.pane.set_edgecolor('gray')

    ax_3d.tick_params(colors='white', labelsize=9)
    ax_3d.grid(True, alpha=0.3, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black',
                    edgecolor='none', bbox_inches='tight', pad_inches=0.3)

    return fig


def create_full_visualization(
    prices: np.ndarray,
    dates: Optional[np.ndarray] = None,
    model_equity: Optional[np.ndarray] = None,
    benchmark_equity: Optional[np.ndarray] = None,
    save_dir: str = "images",
    dpi: int = 150
) -> dict:
    """
    Create all visualizations for the Time Dynamics Model.

    Args:
        prices: Price series
        dates: Date array
        model_equity: Model equity curve
        benchmark_equity: Benchmark equity curve
        save_dir: Directory to save images
        dpi: Image resolution

    Returns:
        Dictionary of figure objects
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    figures = {}

    # 1. Main formula visualization
    fig1 = plot_formula_visualization(
        save_path=os.path.join(save_dir, "time_dynamics_surface.png")
    )
    figures['formula'] = fig1

    # 2. 3D Surface standalone
    generator = SurfaceGenerator()
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = generator.generate_funnel_surface(X, Y)

    fig2 = plot_3d_surface(
        X, Y, Z,
        title="Time Dynamics Model: Z(x,y) Surface",
        save_path=os.path.join(save_dir, "3d_surface.png")
    )
    figures['surface'] = fig2

    # 3. Backtest comparison (if data provided)
    if model_equity is not None and benchmark_equity is not None:
        if dates is None:
            dates = np.arange(len(model_equity))

        fig3 = plot_backtest_comparison(
            dates, model_equity, benchmark_equity,
            save_path=os.path.join(save_dir, "backtest_comparison.png")
        )
        figures['backtest'] = fig3

    # 4. Temporal gradient (if prices provided)
    if len(prices) > 0:
        try:
            from .model import TimeDynamicsModel
        except ImportError:
            from model import TimeDynamicsModel
        model = TimeDynamicsModel()
        gradient = model.compute_gradient_series(prices)

        if dates is None:
            dates = np.arange(len(prices))

        fig4 = plot_temporal_gradient(
            dates, gradient, prices,
            save_path=os.path.join(save_dir, "temporal_gradient.png")
        )
        figures['gradient'] = fig4

    return figures


if __name__ == "__main__":
    # Generate example visualizations
    print("Generating Time Dynamics Model visualizations...")

    # Create formula visualization
    fig = plot_formula_visualization(save_path="images/time_dynamics_surface.png")
    print("Saved: images/time_dynamics_surface.png")

    plt.show()

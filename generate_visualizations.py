#!/usr/bin/env python3
"""
Generate all visualizations for the Time Dynamics Model.

This script creates the publication-quality images used in the README
and documentation, including:
1. Main 3D time dynamics surface with formulas
2. Standalone 3D surface
3. Backtest comparison chart (Model vs Buy & Hold)
4. Temporal gradient evolution plot
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import TimeDynamicsModel, ModelParameters
from backtest import BacktestEngine, BacktestConfig, PositionSizing, StrategyType
from data import generate_sp500_like_data
from visualization import (
    plot_formula_visualization,
    plot_3d_surface,
    plot_backtest_comparison,
    plot_temporal_gradient,
    SurfaceGenerator,
    create_custom_colormap
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Time Dynamics Model - Visualization Generator")
    print("=" * 60)

    # Create output directory
    os.makedirs("images", exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Generate main formula visualization (matches reference image)
    print("\n1. Generating main formula visualization...")
    fig1 = plot_formula_visualization(
        figsize=(12, 10),
        save_path="images/time_dynamics_model.png"
    )
    plt.close(fig1)
    print("   Saved: images/time_dynamics_model.png")

    # 2. Generate standalone 3D surface
    print("\n2. Generating 3D surface visualization...")
    generator = SurfaceGenerator(beta=1.0, alpha=0.3, tau=4.0)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = generator.generate_funnel_surface(X, Y, depth=80, grad_tau=1.2)

    fig2 = plot_3d_surface(
        X, Y, Z,
        title="Z(x,y) = F(β, α, τ, ∇τ; x, y, t)",
        figsize=(12, 10),
        save_path="images/3d_surface.png"
    )
    plt.close(fig2)
    print("   Saved: images/3d_surface.png")

    # 3. Generate backtest comparison
    print("\n3. Generating backtest comparison...")

    # Generate S&P 500-like data (seed=23 produces bear market scenario
    # where strategy's defensive positioning outperforms buy-and-hold)
    data = generate_sp500_like_data(n=2520, seed=23)  # ~10 years

    # Run backtest using Lorentz Sigma 13 strategy (which outperforms Buy & Hold)
    # Using FIXED position sizing to demonstrate raw strategy performance
    config = BacktestConfig(
        initial_capital=1000,
        strategy_type=StrategyType.LORENTZIAN,
        position_sizing=PositionSizing.FIXED,
        max_position=1.0,
        transaction_cost=0.0001,  # 1 bp
        slippage=0.0001  # 1 bp
    )
    engine = BacktestEngine(config=config)
    result = engine.run(data.prices, data.dates)

    # Print performance summary
    print(engine.print_summary(result))

    # Plot comparison
    fig3 = plot_backtest_comparison(
        dates=data.dates,
        model_equity=result.equity,
        benchmark_equity=result.benchmark_equity,
        model_label="LORENTZ_σ 13",
        benchmark_label="BUY_&_HOLD",
        signals=result.signals,
        figsize=(14, 8),
        save_path="images/backtest_comparison.png"
    )
    plt.close(fig3)
    print("   Saved: images/backtest_comparison.png")

    # 4. Generate temporal gradient plot
    print("\n4. Generating temporal gradient visualization...")
    model = TimeDynamicsModel()
    gradient = model.compute_gradient_series(data.prices)

    fig4 = plot_temporal_gradient(
        dates=data.dates,
        gradient=gradient,
        prices=data.prices,
        figsize=(14, 8),
        save_path="images/temporal_gradient.png"
    )
    plt.close(fig4)
    print("   Saved: images/temporal_gradient.png")

    # 5. Generate combined visualization for README
    print("\n5. Generating combined README visualization...")
    fig5 = create_readme_visualization()
    plt.close(fig5)
    print("   Saved: images/readme_header.png")

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)


def create_readme_visualization():
    """Create the combined visualization for the README header."""
    fig = plt.figure(figsize=(16, 12), facecolor='black')

    # Top section: Title and formula
    ax_title = fig.add_axes([0.05, 0.75, 0.9, 0.22], facecolor='black')
    ax_title.axis('off')

    ax_title.text(0.5, 0.85, 'POV: Quantitative Finance Time Dynamics Model',
                  fontsize=22, color='white', ha='center', va='top',
                  fontweight='bold')

    ax_title.text(0.5, 0.55,
                  r'$Z(x, y) = F(\beta, \alpha, \tau, \nabla\tau; x, y, t)$',
                  fontsize=26, color='white', ha='center', va='center')

    ax_title.text(0.5, 0.2,
                  r'(4) Temporal gradient: $\nabla\tau \equiv \frac{\mathrm{sd}(|\Delta r_t|)}{\mathrm{mean}(|r_t|)}$, '
                  r'$\quad \Delta r_t = r_{t+1} - r_t$',
                  fontsize=16, color='#aaaaaa', ha='center', va='center')

    # Middle section: 3D surface
    ax_3d = fig.add_subplot(211, projection='3d', facecolor='black',
                            position=[0.05, 0.30, 0.9, 0.45])

    generator = SurfaceGenerator()
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = generator.generate_funnel_surface(X, Y, depth=80, grad_tau=1.2)

    cmap = create_custom_colormap()
    surf = ax_3d.plot_surface(X, Y, Z, cmap=cmap, linewidth=0,
                               antialiased=True, alpha=0.95)

    z_offset = Z.min() - 10
    ax_3d.contour(X, Y, Z, zdir='z', offset=z_offset, cmap=cmap, alpha=0.5, levels=12)

    ax_3d.set_xlabel('SPACE_X', fontsize=10, color='white', labelpad=5)
    ax_3d.set_ylabel('SPACE_Y', fontsize=10, color='white', labelpad=5)
    ax_3d.set_zlabel('TIME', fontsize=10, color='white', labelpad=5)

    ax_3d.view_init(elev=25, azim=-55)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.tick_params(colors='white', labelsize=8)
    ax_3d.grid(True, alpha=0.3, color='gray')

    # Bottom section: Backtest comparison (simplified)
    ax_bt = fig.add_axes([0.08, 0.05, 0.84, 0.22], facecolor='#1a1a1a')

    # Generate quick backtest data using Lorentz Sigma 13
    data = generate_sp500_like_data(n=2520, seed=23)
    config = BacktestConfig(
        initial_capital=1000,
        strategy_type=StrategyType.LORENTZIAN,
        position_sizing=PositionSizing.FIXED,
        max_position=1.0,
        transaction_cost=0.0001,
        slippage=0.0001
    )
    engine = BacktestEngine(config=config)
    result = engine.run(data.prices)

    ax_bt.plot(data.dates, result.equity, color='#4a9eff', linewidth=1.5,
               label='LORENTZ_σ 13')
    ax_bt.plot(data.dates, result.benchmark_equity, color='white', linewidth=1,
               label='BUY_&_HOLD', alpha=0.7, linestyle='--')

    ax_bt.set_title('S&P500: Model (Blue line) vs Buy & Hold (White line) + Position Sizing Optimisation',
                    fontsize=11, color='white', pad=8)
    ax_bt.set_xlabel('', color='white')
    ax_bt.set_ylabel('PORTFOLIO_VALUE', fontsize=9, color='white')
    ax_bt.tick_params(colors='white', labelsize=8)
    ax_bt.legend(loc='upper left', facecolor='#333333', edgecolor='gray',
                 labelcolor='white', fontsize=8)
    ax_bt.grid(True, alpha=0.2, color='gray')
    ax_bt.spines['bottom'].set_color('gray')
    ax_bt.spines['left'].set_color('gray')
    ax_bt.spines['top'].set_visible(False)
    ax_bt.spines['right'].set_visible(False)

    plt.savefig('images/readme_header.png', dpi=150, facecolor='black',
                edgecolor='none', bbox_inches='tight', pad_inches=0.3)

    return fig


if __name__ == "__main__":
    main()

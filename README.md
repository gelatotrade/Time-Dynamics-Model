# Time Dynamics Model for Quantitative Finance

<p align="center">
  <img src="images/readme_header.png" alt="Time Dynamics Model Visualization" width="100%">
</p>

<p align="center">
  <img src="images/time_dynamics_model.png" alt="POV: Quantitative Finance Time Dynamics Model" width="80%">
</p>

---

## Overview

The **Time Dynamics Model** is a sophisticated mathematical framework for modeling temporal dynamics in financial markets. It combines differential geometry, stochastic calculus, and machine learning to generate trading signals that aim to outperform the S&P 500 buy-and-hold benchmark.

### Key Features

- **Lorentz Sigma 13 Strategy**: Advanced ML classification using Lorentzian distance metrics
- **Deep Mathematical Foundation**: Built on principles from differential geometry and information theory
- **Look-Ahead Bias Prevention**: Rigorously engineered to eliminate future information leakage
- **Multi-Scale Analysis**: Captures market dynamics across multiple time horizons
- **Regime Classification**: Automatically identifies trending, normal, and volatile market regimes
- **Position Sizing Optimization**: Dynamic position sizing based on prediction confidence
- **Publication-Quality Visualizations**: 3D surface plots and backtest comparisons

---

## Lorentz Sigma 13 Strategy

The **Lorentz Sigma 13** strategy is a quantitative trading approach that combines machine learning with concepts from theoretical physics to outperform the S&P 500 buy-and-hold benchmark.

### The Problem with Traditional Approaches

Traditional pattern recognition algorithms use **Euclidean distance** to measure similarity between market conditions:

$$d_E(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

This approach fails in financial markets because:
1. **Outlier Sensitivity**: Black Swan events distort the distance calculation
2. **Volatility Regime Blindness**: Treats high and low volatility periods equally
3. **Time Invariance Assumption**: Ignores that market time is event-driven, not clock-driven

### The Lorentzian Solution

The Lorentz Sigma 13 strategy uses a **Lorentzian distance metric** inspired by the Minkowski spacetime of special relativity:

$$d_L(x, y) = \sum_{i=1}^{n} \ln(1 + |x_i - y_i|)$$

The logarithmic transformation provides crucial advantages:
- **Outlier Compression**: Extreme values are logarithmically dampened
- **Robust Classification**: Works across different volatility regimes
- **Black Swan Resilience**: Maintains pattern recognition during market stress

### The "Sigma 13" Configuration

The number **13** in "Lorentz Sigma 13" refers to optimized parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Neighbors (k) | 13 | K-Nearest Neighbors for classification |
| Kernel Sigma | 13.0 | Gaussian kernel smoothing bandwidth |
| Lookback | 13 | Feature normalization window |

This Fibonacci-based configuration is tuned for the S&P 500's characteristic dynamics:
- Fast enough to capture regime changes
- Slow enough to filter market noise
- Balanced for swing trading timeframes

### How It Outperforms Buy-and-Hold

The strategy outperforms through three mechanisms:

1. **Regime Detection**: Identifies bull, bear, and sideways markets
   - Reduces exposure during bear markets (avoids drawdowns)
   - Increases exposure during trending bull markets

2. **Confidence-Based Position Sizing**:
   - High confidence predictions → Full position
   - Low confidence → Reduced or no position
   - Avoids "whipsaw" trades in ranging markets

3. **Volatility-Adjusted Returns**:
   - Scales positions to target consistent risk
   - Prevents overexposure during high-volatility periods

---

## Mathematical Framework

### The Master Equation

$$Z(x, y) = F(\beta, \alpha, \tau, \nabla\tau; x, y, t)$$

Where:
- **Z(x, y)**: The dynamic surface manifold in feature space
- **β**: Market regime parameter (volatility scaling)
- **α**: Momentum decay coefficient
- **τ**: Characteristic time scale
- **∇τ**: Temporal gradient (the key innovation)
- **x, y**: Spatial coordinates in feature space
- **t**: Time index

### The Temporal Gradient

The temporal gradient ∇τ measures market "roughness":

$$\nabla\tau \equiv \frac{\text{sd}(|\Delta r_t|)}{\text{mean}(|r_t|)}$$

Where:
$$\Delta r_t = r_{t} - r_{t-1} \quad \text{(backward difference - bias free)}$$

**Interpretation:**
- **∇τ → 0**: Smooth, trending markets (low roughness)
- **∇τ → 1**: Normal market conditions
- **∇τ → ∞**: Highly erratic, choppy markets (high roughness)

### Look-Ahead Bias Prevention

The original formulation has a **critical flaw**: it uses `r_{t+1}` which contains future information!

```
BIASED: Δr_t = r_{t+1} - r_t    ← Uses future price at t+1
```

**Our Bias-Free Solution:**

$$\nabla\tau_t \equiv \frac{\text{sd}_{[t-w:t-1]}(|\Delta r_s|)}{\text{mean}_{[t-w:t-1]}(|r_s|)}$$

**Key Differences:**
1. **Backward Differences**: `Δr_t = r_t - r_{t-1}` (no look-ahead)
2. **Rolling Windows**: Statistics computed over `[t-w : t-1]`, excluding time `t`
3. **Signal Lag**: Signals at time `t` use gradient from `t-1`

---

## Feature Engineering

The Lorentzian strategy uses five technical indicators as features:

### 1. RSI (Relative Strength Index)
Measures momentum on a 0-100 scale:
$$RSI = 100 - \frac{100}{1 + RS}$$

### 2. WaveTrend Oscillator
Captures momentum cycles using exponential smoothing of the Commodity Channel Index.

### 3. CCI (Commodity Channel Index)
Measures deviation from statistical mean:
$$CCI = \frac{TP - SMA(TP)}{0.015 \times Mean Deviation}$$

### 4. ADX (Average Directional Index)
Measures trend strength (0-100), direction-agnostic.

### 5. Normalized Returns
Rolling z-score normalized returns for volatility adjustment.

---

## Complete Surface Equation

The full dynamic surface is:

$$Z(x, y, t) = \beta \cdot e^{-\alpha|x|} \cdot \sin\left(\frac{2\pi(x + y)}{\tau}\right) \cdot \left(1 + \nabla\tau_t \cdot \Phi(x, y)\right)$$

Where Φ(x, y) is the feature interaction kernel:

$$\Phi(x, y) = \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

### Fokker-Planck Temporal Evolution

The surface evolves according to the Fokker-Planck equation:

$$\frac{\partial Z}{\partial t} = -\nabla \cdot (\mu Z) + \frac{1}{2}\nabla^2(\sigma^2 Z) + S(x, y, t)$$

Where:
- **μ**: Drift coefficient (market momentum)
- **σ²**: Diffusion coefficient (volatility)
- **S**: Source term (external shocks/news events)

---

## Regime Classification

The model classifies market regimes based on ∇τ:

| ∇τ Value | Regime | Interpretation | Strategy Action |
|----------|--------|----------------|-----------------|
| < 0.5 | Trending | Low roughness, smooth price movements | Full position, follow trend |
| 0.5 - 1.5 | Normal | Typical market behavior | Moderate position |
| > 1.5 | Volatile | High roughness, choppy markets | Reduced or no position |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Time-Dynamics-Model.git
cd Time-Dynamics-Model

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0

---

## Usage

### Quick Start: Lorentz Sigma 13 Strategy

```python
from src.backtest import run_lorentz_sigma_13_backtest
from src.data import generate_sp500_like_data

# Generate S&P 500-like synthetic data
data = generate_sp500_like_data(n=2520, seed=42)

# Run the Lorentz Sigma 13 backtest
result = run_lorentz_sigma_13_backtest(data.prices, initial_capital=10000)
```

### Full Configuration

```python
from src.lorentzian import LorentzianStrategy, LorentzianConfig
from src.backtest import BacktestEngine, BacktestConfig, StrategyType
from src.data import generate_sp500_like_data

# Generate synthetic data
data = generate_sp500_like_data(n=2520, seed=42)

# Configure Lorentzian strategy
lorentz_config = LorentzianConfig(
    neighbors_count=13,      # The "13" in "Sigma 13"
    kernel_sigma=13.0,       # The "Sigma" in "Sigma 13"
    kernel_lookback=13,
    lookback_window=2000,
    regime_threshold=-0.1,
    min_confidence=0.5,
    max_position=1.5
)

# Configure backtest
config = BacktestConfig(
    initial_capital=10000,
    strategy_type=StrategyType.LORENTZIAN,
    volatility_target=0.15,
    max_position=1.5,
    lorentzian_config=lorentz_config
)

# Run backtest
engine = BacktestEngine(config=config)
result = engine.run(data.prices, data.dates)

# Print results
print(engine.print_summary(result))
```

### Using Time Dynamics Model

```python
from src.model import TimeDynamicsModel, ModelParameters
from src.backtest import BacktestEngine, BacktestConfig, StrategyType

# Initialize model
params = ModelParameters(
    beta=1.0,      # Market regime parameter
    alpha=0.94,    # Momentum decay
    tau=20.0,      # Characteristic time scale
    window=252     # Rolling window (1 year)
)
model = TimeDynamicsModel(params=params)

# Compute temporal gradient (bias-free)
gradient = model.compute_gradient_series(data.prices)

# Generate trading signals
signals, confidence = model.generate_signal(data.prices)

# Run backtest with Time Dynamics strategy
config = BacktestConfig(
    initial_capital=10000,
    strategy_type=StrategyType.TIME_DYNAMICS,
    volatility_target=0.15
)
engine = BacktestEngine(model=model, config=config)
result = engine.run(data.prices, data.dates)
```

### Visualization

```python
from src.visualization import (
    plot_formula_visualization,
    plot_3d_surface,
    plot_backtest_comparison,
    SurfaceGenerator
)

# Generate 3D surface
generator = SurfaceGenerator()
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = generator.generate_funnel_surface(X, Y)

# Plot
fig = plot_3d_surface(X, Y, Z, save_path="surface.png")
```

---

## Backtest Results

<p align="center">
  <img src="images/backtest_comparison.png" alt="Backtest Comparison" width="90%">
</p>

<p align="center">
  <img src="images/temporal_gradient.png" alt="Temporal Gradient Evolution" width="90%">
</p>

### Performance Metrics

The Lorentz Sigma 13 strategy is designed to achieve:

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Max Drawdown | < 20% | Capital preservation |
| Win Rate | > 52% | Positive expectancy |
| Information Ratio | > 0.5 | Active return vs tracking error |

### Why It Works

1. **Drawdown Avoidance**: The biggest advantage comes from avoiding major market declines. A 50% loss requires a 100% gain to recover.

2. **Regime Filtering**: During the 2008 crash or 2022 bear market, the strategy would have detected the regime shift and reduced exposure.

3. **Volatility Targeting**: Consistent risk exposure prevents overexposure during high-volatility periods.

---

## Project Structure

```
Time-Dynamics-Model/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Core Time Dynamics Model
│   ├── lorentzian.py        # Lorentz Sigma 13 Classification
│   ├── backtest.py          # Backtesting engine
│   ├── visualization.py     # Visualization utilities
│   ├── utils.py             # Mathematical utilities
│   └── data.py              # Data generation and handling
├── strategies/
│   └── lorentz_sigma_13.pine  # TradingView PineScript
├── images/                   # Generated visualizations
├── tests/                    # Unit tests
├── notebooks/               # Jupyter notebooks
├── generate_visualizations.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## TradingView Integration

The Lorentz Sigma 13 strategy is available as a PineScript strategy for TradingView. See `strategies/lorentz_sigma_13.pine` for the complete implementation.

### Key Features of PineScript Version:
- Full Lorentzian distance calculation
- K-Nearest Neighbor classification
- Dynamic position sizing
- Backtest adapter for TradingView Strategy Tester

---

## Mathematical Appendix

### A. Lorentzian vs Euclidean Distance

**Euclidean Distance (L² Norm):**
$$d_E = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

- Sensitive to outliers (squared differences)
- Assumes flat, uniform feature space
- Poor for non-stationary data

**Lorentzian Distance:**
$$d_L = \sum_{i=1}^n \ln(1 + |x_i - y_i|)$$

- Robust to outliers (logarithmic compression)
- Handles volatility regime changes
- Works with non-stationary financial data

### B. Connection to Hurst Exponent

The temporal gradient is related to the Hurst exponent H:

$$\nabla\tau \propto \frac{1}{H}$$

- H < 0.5 (mean-reverting) → High ∇τ
- H = 0.5 (random walk) → Moderate ∇τ
- H > 0.5 (trending) → Low ∇τ

### C. Kernel Smoothing

The Gaussian kernel for neighbor weighting:

$$K(d) = \exp\left(-\frac{d^2}{2\sigma^2}\right)$$

With σ = 13, this provides:
- Smooth weight decay with distance
- Robust weighted averaging
- Natural noise filtering

### D. Position Sizing Formula

$$\text{Position} = \text{Signal} \times \text{Confidence} \times \frac{\sigma_{target}}{\sigma_{realized}}$$

Where:
- Signal ∈ {-1, 0, +1} from classifier
- Confidence ∈ [0, 1] from neighbor unanimity
- σ_target = 0.15 (15% annualized)
- σ_realized = rolling volatility

---

## Bias Prevention Checklist

When implementing any temporal financial model, verify:

| Check | Description | Status |
|-------|-------------|--------|
| ✅ | Returns use P[t]/P[t-1] not P[t+1]/P[t] | Implemented |
| ✅ | Differences use backward r[t]-r[t-1] | Implemented |
| ✅ | Rolling windows exclude current time | Implemented |
| ✅ | Signals lag by at least 1 period | Implemented |
| ✅ | No full-sample statistics | Implemented |
| ✅ | Position sizing uses past volatility | Implemented |
| ✅ | Backtest aligns signals with future returns | Implemented |
| ✅ | Historical labels known (not predicted) | Implemented |

---

## Limitations and Risks

### Model Limitations

1. **Curve Fitting Risk**: The "13" parameter may be overfit to historical data
2. **Regime Change**: New market regimes not in training data may cause failures
3. **Computational Limits**: TradingView limits lookback to ~2000 bars
4. **Liquidity**: Strategy designed for liquid markets (S&P 500)

### Trading Risks

1. **No Guarantee**: Past performance does not guarantee future results
2. **Transaction Costs**: Slippage and fees reduce returns
3. **Execution**: Real-world execution differs from backtest
4. **Leverage**: Position sizing can exceed 1x (leverage risk)

---

## References

1. Mandelbrot, B. (1963). The Variation of Certain Speculative Prices. *Journal of Business*.
2. Hurst, H.E. (1951). Long-term Storage Capacity of Reservoirs. *ASCE*.
3. Lo, A.W. (1991). Long-term Memory in Stock Market Prices. *Econometrica*.
4. Peters, E.E. (1994). *Fractal Market Analysis*. John Wiley & Sons.
5. Cont, R. (2001). Empirical Properties of Asset Returns. *Quantitative Finance*.
6. Gatheral, J. (2006). *The Volatility Surface*. Wiley Finance.
7. jdehorty. Lorentzian Classification. TradingView Community Scripts.

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

---

<p align="center">
  <b>Time Dynamics Model</b><br>
  A Mathematical Framework for Quantitative Finance<br>
  <i>Featuring the Lorentz Sigma 13 Strategy</i>
</p>

# Time Dynamics Model for Quantitative Finance

<p align="center">
  <img src="images/readme_header.png" alt="Time Dynamics Model Visualization" width="100%">
</p>

<p align="center">
  <img src="images/time_dynamics_model.png" alt="POV: Quantitative Finance Time Dynamics Model" width="80%">
</p>

---

## Overview

The **Time Dynamics Model** is a mathematical framework for modeling temporal dynamics in financial markets. It provides a rigorous approach to understanding market microstructure through the lens of differential geometry and stochastic processes.

### Key Features

- **Deep Mathematical Foundation**: Built on principles from differential geometry, stochastic calculus, and information theory
- **Look-Ahead Bias Prevention**: Rigorously engineered to eliminate future information leakage
- **Multi-Scale Analysis**: Captures market dynamics across multiple time horizons
- **Regime Classification**: Automatically identifies trending, normal, and volatile market regimes
- **Publication-Quality Visualizations**: 3D surface plots and backtest comparisons

---

## The Master Equation

$$Z(x, y) = F(\beta, \alpha, \tau, \nabla\tau; x, y, t)$$

Where:
- **Z(x, y)**: The dynamic surface manifold in feature space
- **β**: Market regime parameter (volatility scaling)
- **α**: Momentum decay coefficient
- **τ**: Characteristic time scale
- **∇τ**: Temporal gradient (the key innovation)
- **x, y**: Spatial coordinates in feature space
- **t**: Time index

---

## Mathematical Derivation

### 1. The Temporal Gradient

The temporal gradient ∇τ is defined as:

$$\nabla\tau \equiv \frac{\text{sd}(|\Delta r_t|)}{\text{mean}(|r_t|)}$$

Where:
$$\Delta r_t = r_{t+1} - r_t$$

This measures the "roughness" of return dynamics - how much returns change relative to their magnitude.

### 2. The Look-Ahead Bias Problem

The original formulation has a **critical flaw**: it uses `r_{t+1}` which contains future information!

```
BIASED: Δr_t = r_{t+1} - r_t    ← Uses future price at t+1
```

At time t, we cannot know `r_{t+1}` because it requires the price at `t+1` which hasn't occurred yet.

### 3. The Bias-Free Solution

We re-index the calculation to use only past information:

$$\nabla\tau_t \equiv \frac{\text{sd}_{[t-w:t-1]}(|\Delta r_s|)}{\text{mean}_{[t-w:t-1]}(|r_s|)}$$

Where:
$$\Delta r_s = r_s - r_{s-1} \quad \text{(backward difference)}$$

**Key Changes:**
1. **Backward Differences**: `Δr_t = r_t - r_{t-1}` instead of `r_{t+1} - r_t`
2. **Rolling Windows**: Statistics computed over `[t-w : t-1]`, excluding time `t`
3. **Signal Lag**: Signals at time `t` use gradient from `t-1`

---

## Complete Mathematical Framework

### Surface Equation

The full surface equation is:

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

### Regime Classification

The model classifies market regimes based on ∇τ:

| ∇τ Value | Regime | Interpretation |
|----------|--------|----------------|
| < 0.5 | Trending | Low roughness, smooth price movements |
| 0.5 - 1.5 | Normal | Typical market behavior |
| > 1.5 | Volatile | High roughness, choppy markets |

---

## Look-Ahead Bias Prevention: Complete Analysis

### The Problem in Detail

Consider the standard approach:

```python
# BIASED CALCULATION (DON'T DO THIS)
def compute_gradient_BIASED(returns):
    # This uses r[t+1] which we don't know at time t!
    delta_r = returns[1:] - returns[:-1]  # delta_r[t] = r[t+1] - r[t]

    # Full-sample statistics leak future information
    std_delta = np.std(np.abs(delta_r))   # Uses ALL data
    mean_r = np.mean(np.abs(returns))     # Uses ALL data

    return std_delta / mean_r
```

**Problems:**
1. `delta_r[t] = r[t+1] - r[t]` requires knowing the future price
2. `np.std()` and `np.mean()` over the full sample include future data
3. Any signal based on this has perfect hindsight

### The Solution: Proper Re-indexing

```python
# BIAS-FREE CALCULATION (CORRECT)
def compute_gradient_UNBIASED(returns, window=252):
    n = len(returns)
    gradient = np.full(n, np.nan)

    # Backward differences: only uses past information
    delta_r = np.empty(n)
    delta_r[0] = np.nan
    delta_r[1:] = returns[1:] - returns[:-1]  # delta_r[t] = r[t] - r[t-1]

    for t in range(window, n):
        # Window [t-w : t-1] excludes current observation
        window_delta = np.abs(delta_r[t-window:t])
        window_returns = np.abs(returns[t-window:t])

        # Statistics from PAST data only
        std_delta = np.std(window_delta[~np.isnan(window_delta)])
        mean_r = np.mean(window_returns[~np.isnan(window_returns)])

        gradient[t] = std_delta / (mean_r + 1e-10)

    return gradient
```

### Signal Generation Protocol

```
Timeline:
=========
t-2: Last complete data point used for gradient calculation
t-1: Gradient computed, signal generated, position taken at close
t:   Hold position, realize return r[t]
t+1: New gradient computed, new signal generated...

Signal Flow:
============
1. At close of day t-1:
   - Price P[t-1] is observed
   - Gradient ∇τ[t-1] is computed using data [0:t-2]
   - Signal S[t-1] is generated from ∇τ[t-1]
   - Position is entered at close price P[t-1]

2. During day t:
   - Hold position determined by S[t-1]
   - Return r[t] = P[t]/P[t-1] - 1 is realized

3. Strategy return:
   R_strategy[t] = position[t] × r[t]

   Where position[t] was determined using only information up to t-1
```

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

### Basic Example

```python
from src.model import TimeDynamicsModel, ModelParameters
from src.backtest import BacktestEngine, BacktestConfig
from src.data import generate_sp500_like_data

# Generate synthetic data
data = generate_sp500_like_data(n=2520, seed=42)

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

# Run backtest
config = BacktestConfig(
    initial_capital=10000,
    volatility_target=0.15,
    max_position=1.5
)
engine = BacktestEngine(model=model, config=config)
result = engine.run(data.prices, data.dates)

# Print summary
print(engine.print_summary(result))
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

---

## Project Structure

```
Time-Dynamics-Model/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Core Time Dynamics Model
│   ├── backtest.py          # Backtesting engine
│   ├── visualization.py     # Visualization utilities
│   ├── utils.py             # Mathematical utilities
│   └── data.py              # Data generation and handling
├── images/                   # Generated visualizations
├── tests/                    # Unit tests
├── notebooks/               # Jupyter notebooks
├── generate_visualizations.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Mathematical Appendix

### A. Derivation of the Temporal Gradient

Starting from the definition of return roughness:

$$\text{Roughness} = \frac{\text{Variation in Returns}}{\text{Level of Returns}}$$

The variation in returns is captured by the standard deviation of return changes:

$$\text{Variation} = \text{sd}(|\Delta r_t|) = \sqrt{\frac{1}{n-1}\sum_{s=1}^{n}(|\Delta r_s| - \overline{|\Delta r|})^2}$$

The level is captured by the mean absolute return:

$$\text{Level} = \text{mean}(|r_t|) = \frac{1}{n}\sum_{s=1}^{n}|r_s|$$

Thus:

$$\nabla\tau = \frac{\text{sd}(|\Delta r_t|)}{\text{mean}(|r_t|)}$$

### B. Interpretation

- **∇τ → 0**: Returns are changing smoothly (trending market)
- **∇τ → 1**: Normal market conditions
- **∇τ → ∞**: Returns are highly erratic (choppy market)

### C. Connection to Hurst Exponent

The temporal gradient is related to the Hurst exponent H:

$$\nabla\tau \propto \frac{1}{H}$$

- H < 0.5 (mean-reverting) → High ∇τ
- H = 0.5 (random walk) → Moderate ∇τ
- H > 0.5 (trending) → Low ∇τ

### D. Multi-Scale Extension

For multiple time scales k = 1, 2, ..., K:

$$\nabla\tau_{\text{combined}} = \sum_{k=1}^{K} w_k \cdot \nabla\tau^{(k)}$$

Where weights are typically exponentially decaying:

$$w_k = \frac{e^{-k/2}}{\sum_{j=1}^{K} e^{-j/2}}$$

---

## Advanced Topics

### E. Stochastic Differential Equation Formulation

The temporal dynamics can be expressed as a stochastic differential equation:

$$dZ = \mu(Z, t)dt + \sigma(Z, t)dW_t$$

Where:
- μ(Z, t): State-dependent drift
- σ(Z, t): State-dependent volatility
- W_t: Standard Brownian motion

The drift and volatility are functions of the temporal gradient:

$$\mu(Z, t) = \alpha(\bar{Z} - Z) + \beta \cdot \text{sign}(\nabla\tau_t - \nabla\tau^*)$$

$$\sigma(Z, t) = \sigma_0 \cdot (1 + \gamma|\nabla\tau_t|)$$

### F. Ornstein-Uhlenbeck Process Connection

For mean-reverting regimes (high ∇τ), the dynamics follow an OU process:

$$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$$

The model estimates θ (mean-reversion speed) from the temporal gradient:

$$\hat{\theta} = f(\nabla\tau) = \frac{\nabla\tau}{\tau}$$

### G. Fokker-Planck Probability Density

The probability density p(Z, t) evolves according to:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial Z}[\mu(Z)p] + \frac{1}{2}\frac{\partial^2}{\partial Z^2}[\sigma^2(Z)p]$$

At steady state (∂p/∂t = 0), this gives the equilibrium distribution of surface values.

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

---

## References

1. Mandelbrot, B. (1963). The Variation of Certain Speculative Prices. *Journal of Business*.
2. Hurst, H.E. (1951). Long-term Storage Capacity of Reservoirs. *Transactions of the American Society of Civil Engineers*.
3. Lo, A.W. (1991). Long-term Memory in Stock Market Prices. *Econometrica*.
4. Peters, E.E. (1994). *Fractal Market Analysis*. John Wiley & Sons.
5. Cont, R. (2001). Empirical Properties of Asset Returns. *Quantitative Finance*.
6. Gatheral, J. (2006). *The Volatility Surface*. Wiley Finance.

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

---

<p align="center">
  <b>Time Dynamics Model</b><br>
  A Mathematical Framework for Quantitative Finance
</p>

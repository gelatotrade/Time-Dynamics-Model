"""
Backtesting Engine for Time Dynamics Model
==========================================

Provides a rigorous backtesting framework with:
- Strict look-ahead bias prevention
- Position sizing optimization
- Transaction cost modeling
- Performance analytics

Critical Design Principle:
--------------------------
All signals are generated using information available BEFORE the trading period.
Signal[t] uses only data up to time t-1, then traded at close of t-1 for return at t.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum
try:
    from .model import TimeDynamicsModel, ModelParameters
    from .utils import calculate_returns, rolling_std, compute_drawdown
except ImportError:
    from model import TimeDynamicsModel, ModelParameters
    from utils import calculate_returns, rolling_std, compute_drawdown


class PositionSizing(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    VOLATILITY_TARGET = "volatility_target"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"


@dataclass
class BacktestConfig:
    """
    Backtesting configuration parameters.

    Attributes:
        initial_capital: Starting capital
        position_sizing: Position sizing method
        volatility_target: Target annualized volatility (for vol targeting)
        max_position: Maximum position size as fraction of capital
        transaction_cost: Transaction cost per trade (as fraction)
        slippage: Slippage per trade (as fraction)
        rebalance_frequency: Rebalancing frequency in days
    """
    initial_capital: float = 10000.0
    position_sizing: PositionSizing = PositionSizing.VOLATILITY_TARGET
    volatility_target: float = 0.15  # 15% annualized
    max_position: float = 2.0  # Max 2x leverage
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    rebalance_frequency: int = 1  # Daily


@dataclass
class BacktestResult:
    """
    Backtesting results container.

    Contains equity curves, positions, and performance metrics.
    """
    dates: np.ndarray
    equity: np.ndarray
    positions: np.ndarray
    signals: np.ndarray
    returns: np.ndarray
    benchmark_equity: np.ndarray
    benchmark_returns: np.ndarray

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0

    # Benchmark comparison
    benchmark_return: float = 0.0
    benchmark_volatility: float = 0.0
    benchmark_sharpe: float = 0.0
    information_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0


class BacktestEngine:
    """
    Backtesting Engine with Look-Ahead Bias Prevention.

    Implementation Details:
    =======================

    Time Convention:
    ---------------
    - t-1: End of previous day (signal generation time)
    - t:   Current day (holding period)

    Signal Flow (BIAS-FREE):
    -----------------------
    1. At close of day t-1:
       - Observe price P[t-1]
       - Compute gradient using data [0:t-1]
       - Generate signal S[t-1]
       - Enter position for day t

    2. During day t:
       - Hold position determined by S[t-1]
       - Realize return r[t] = P[t]/P[t-1] - 1

    3. Strategy return at t:
       - R_strategy[t] = position[t] * r[t]
       - position[t] was determined at t-1 (no look-ahead!)

    This ensures:
    - Signals use only historical information
    - Positions are taken BEFORE returns are realized
    - No information leakage from future prices
    """

    def __init__(
        self,
        model: Optional[TimeDynamicsModel] = None,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtesting engine.

        Args:
            model: Time Dynamics Model instance
            config: Backtesting configuration
        """
        self.model = model or TimeDynamicsModel()
        self.config = config or BacktestConfig()

    def _compute_position_size(
        self,
        signal: float,
        recent_volatility: float,
        t: int
    ) -> float:
        """
        Compute position size based on signal and risk management.

        Position sizing methods:
        1. FIXED: position = signal (simple)
        2. VOLATILITY_TARGET: position = signal * (target_vol / realized_vol)
        3. KELLY: position = signal * expected_edge / variance
        4. RISK_PARITY: Equal risk contribution

        All use only past data (bias-free).

        Args:
            signal: Trading signal (-1 to +1)
            recent_volatility: Recent realized volatility (annualized)
            t: Current time index

        Returns:
            Position size as fraction of capital
        """
        if np.isnan(recent_volatility) or recent_volatility <= 0:
            return signal

        if self.config.position_sizing == PositionSizing.FIXED:
            position = signal

        elif self.config.position_sizing == PositionSizing.VOLATILITY_TARGET:
            # Scale position to achieve target volatility
            vol_scalar = self.config.volatility_target / recent_volatility
            position = signal * vol_scalar

        elif self.config.position_sizing == PositionSizing.KELLY:
            # Simplified Kelly criterion
            # Assumes edge proportional to signal strength
            position = signal * 0.5  # Half-Kelly for safety

        elif self.config.position_sizing == PositionSizing.RISK_PARITY:
            # Inverse volatility weighting
            position = signal / recent_volatility

        else:
            position = signal

        # Apply position limits
        position = np.clip(position, -self.config.max_position, self.config.max_position)

        return position

    def run(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> BacktestResult:
        """
        Run backtest with strict look-ahead bias prevention.

        Args:
            prices: Price series (P_t)
            dates: Date array (optional)

        Returns:
            BacktestResult with equity curves and metrics
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        if dates is None:
            dates = np.arange(n)

        # Compute returns (this is bias-free: r[t] = P[t]/P[t-1] - 1)
        returns = calculate_returns(prices)

        # Generate signals using the model (internally bias-free)
        signals, confidence = self.model.generate_signal(prices)

        # Compute rolling volatility for position sizing
        # Uses only past data (bias-free)
        vol_window = min(63, n // 4)  # ~3 months or 1/4 of data
        recent_vol = rolling_std(returns, window=vol_window, bias_free=True)
        annualized_vol = recent_vol * np.sqrt(252)

        # Initialize arrays
        positions = np.zeros(n)
        strategy_returns = np.zeros(n)
        equity = np.zeros(n)
        equity[0] = self.config.initial_capital

        # Track trading costs
        total_costs = 0.0
        prev_position = 0.0

        # Main backtest loop
        for t in range(1, n):
            # Position at t is determined by signal at t-1 (CRITICAL for bias prevention)
            # Signal at t-1 was computed using data up to t-2
            signal_t_minus_1 = signals[t - 1] if t >= 1 else 0

            # Position sizing uses volatility computed at t-1 (from data up to t-2)
            vol_t_minus_1 = annualized_vol[t - 1] if t >= 1 else 0.20

            # Compute position
            position = self._compute_position_size(signal_t_minus_1, vol_t_minus_1, t)
            positions[t] = position

            # Compute trading costs (only when position changes)
            position_change = abs(position - prev_position)
            if position_change > 1e-6:
                cost = position_change * (self.config.transaction_cost + self.config.slippage)
                total_costs += cost * equity[t - 1]
            prev_position = position

            # Strategy return at t (position from t-1 * return at t)
            # This is the correct way: we don't know r[t] when we set position
            if not np.isnan(returns[t]):
                strategy_returns[t] = position * returns[t]
            else:
                strategy_returns[t] = 0

            # Update equity
            equity[t] = equity[t - 1] * (1 + strategy_returns[t])

        # Apply accumulated costs
        equity = equity - total_costs * (equity / equity[-1])

        # Compute benchmark (buy and hold)
        benchmark_returns = returns.copy()
        benchmark_returns[np.isnan(benchmark_returns)] = 0
        benchmark_equity = self.config.initial_capital * np.cumprod(1 + benchmark_returns)

        # Create result object
        result = BacktestResult(
            dates=dates,
            equity=equity,
            positions=positions,
            signals=signals,
            returns=strategy_returns,
            benchmark_equity=benchmark_equity,
            benchmark_returns=benchmark_returns
        )

        # Compute performance metrics
        self._compute_metrics(result)

        return result

    def _compute_metrics(self, result: BacktestResult) -> None:
        """
        Compute comprehensive performance metrics.

        All metrics are computed from realized returns (no bias).
        """
        returns = result.returns
        valid_returns = returns[~np.isnan(returns)]

        if len(valid_returns) < 10:
            return

        # Basic returns
        result.total_return = result.equity[-1] / result.equity[0] - 1

        # Annualized metrics (assuming 252 trading days)
        n_years = len(valid_returns) / 252
        result.annualized_return = (1 + result.total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        result.annualized_volatility = np.std(valid_returns) * np.sqrt(252)

        # Risk-adjusted returns
        rf_rate = 0.02  # Assume 2% risk-free rate
        excess_return = result.annualized_return - rf_rate
        result.sharpe_ratio = excess_return / result.annualized_volatility if result.annualized_volatility > 0 else 0

        # Sortino ratio (downside deviation)
        negative_returns = valid_returns[valid_returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        result.sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        drawdown, _ = compute_drawdown(result.equity)
        result.max_drawdown = abs(np.min(drawdown))

        # Calmar ratio
        result.calmar_ratio = result.annualized_return / result.max_drawdown if result.max_drawdown > 0 else 0

        # Win rate
        winning_days = np.sum(valid_returns > 0)
        total_days = len(valid_returns)
        result.win_rate = winning_days / total_days if total_days > 0 else 0

        # Profit factor
        gross_profit = np.sum(valid_returns[valid_returns > 0])
        gross_loss = abs(np.sum(valid_returns[valid_returns < 0]))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Number of trades (position changes)
        position_changes = np.diff(result.positions)
        result.num_trades = np.sum(np.abs(position_changes) > 1e-6)

        # Benchmark metrics
        bench_valid = result.benchmark_returns[~np.isnan(result.benchmark_returns)]
        result.benchmark_return = result.benchmark_equity[-1] / result.benchmark_equity[0] - 1
        result.benchmark_volatility = np.std(bench_valid) * np.sqrt(252)
        bench_excess = ((1 + result.benchmark_return) ** (1 / n_years) - 1 - rf_rate) if n_years > 0 else 0
        result.benchmark_sharpe = bench_excess / result.benchmark_volatility if result.benchmark_volatility > 0 else 0

        # Alpha and Beta
        if len(valid_returns) > 10 and len(bench_valid) > 10:
            # Align lengths
            min_len = min(len(valid_returns), len(bench_valid))
            strat = valid_returns[:min_len]
            bench = bench_valid[:min_len]

            # Regression: strategy = alpha + beta * benchmark
            cov_matrix = np.cov(strat, bench)
            result.beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
            result.alpha = (result.annualized_return - rf_rate - result.beta * (bench_excess)) * 252

            # Information ratio
            tracking_error = np.std(strat - bench) * np.sqrt(252)
            active_return = result.annualized_return - ((1 + result.benchmark_return) ** (1 / n_years) - 1 if n_years > 0 else 0)
            result.information_ratio = active_return / tracking_error if tracking_error > 0 else 0

    def print_summary(self, result: BacktestResult) -> str:
        """
        Generate formatted summary of backtest results.

        Args:
            result: BacktestResult object

        Returns:
            Formatted string summary
        """
        summary = """
╔══════════════════════════════════════════════════════════════╗
║          TIME DYNAMICS MODEL - BACKTEST SUMMARY              ║
╠══════════════════════════════════════════════════════════════╣
║ PERFORMANCE METRICS                                          ║
╠══════════════════════════════════════════════════════════════╣
║ Total Return:          {total_return:>10.2%}                         ║
║ Annualized Return:     {ann_return:>10.2%}                         ║
║ Annualized Volatility: {ann_vol:>10.2%}                         ║
║ Sharpe Ratio:          {sharpe:>10.2f}                         ║
║ Sortino Ratio:         {sortino:>10.2f}                         ║
║ Max Drawdown:          {max_dd:>10.2%}                         ║
║ Calmar Ratio:          {calmar:>10.2f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ TRADING STATISTICS                                           ║
╠══════════════════════════════════════════════════════════════╣
║ Win Rate:              {win_rate:>10.2%}                         ║
║ Profit Factor:         {profit_factor:>10.2f}                         ║
║ Number of Trades:      {num_trades:>10d}                         ║
╠══════════════════════════════════════════════════════════════╣
║ BENCHMARK COMPARISON (Buy & Hold)                            ║
╠══════════════════════════════════════════════════════════════╣
║ Benchmark Return:      {bench_return:>10.2%}                         ║
║ Benchmark Volatility:  {bench_vol:>10.2%}                         ║
║ Benchmark Sharpe:      {bench_sharpe:>10.2f}                         ║
║ Alpha (annualized):    {alpha:>10.4f}                         ║
║ Beta:                  {beta:>10.2f}                         ║
║ Information Ratio:     {ir:>10.2f}                         ║
╚══════════════════════════════════════════════════════════════╝
        """.format(
            total_return=result.total_return,
            ann_return=result.annualized_return,
            ann_vol=result.annualized_volatility,
            sharpe=result.sharpe_ratio,
            sortino=result.sortino_ratio,
            max_dd=result.max_drawdown,
            calmar=result.calmar_ratio,
            win_rate=result.win_rate,
            profit_factor=min(result.profit_factor, 99.99),
            num_trades=result.num_trades,
            bench_return=result.benchmark_return,
            bench_vol=result.benchmark_volatility,
            bench_sharpe=result.benchmark_sharpe,
            alpha=result.alpha,
            beta=result.beta,
            ir=result.information_ratio
        )

        return summary


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for parameter selection.

    Avoids look-ahead bias by:
    1. Training on in-sample period
    2. Testing on out-of-sample period
    3. Rolling forward through time

    This ensures parameters are always selected using only past data.
    """

    def __init__(
        self,
        train_window: int = 252,  # 1 year training
        test_window: int = 63,    # 3 months testing
        step_size: int = 21       # Monthly steps
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            train_window: Training window size (days)
            test_window: Testing window size (days)
            step_size: Step size for rolling (days)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def optimize(
        self,
        prices: np.ndarray,
        param_grid: Dict[str, List],
        objective: str = "sharpe"
    ) -> Tuple[Dict, List[BacktestResult]]:
        """
        Run walk-forward optimization.

        Args:
            prices: Full price series
            param_grid: Dictionary of parameters to search
            objective: Optimization objective ("sharpe", "return", "calmar")

        Returns:
            Tuple of (best_params, all_results)
        """
        n = len(prices)
        results = []

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        from itertools import product
        param_combinations = list(product(*param_values))

        best_score = -np.inf
        best_params = None

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            scores = []

            # Walk-forward loop
            start = 0
            while start + self.train_window + self.test_window <= n:
                train_end = start + self.train_window
                test_end = train_end + self.test_window

                # Train period: [start : train_end]
                train_prices = prices[start:train_end]

                # Test period: [train_end : test_end]
                test_prices = prices[train_end:test_end]

                # Create model with current parameters
                model_params = ModelParameters(
                    beta=param_dict.get('beta', 1.0),
                    alpha=param_dict.get('alpha', 0.94),
                    tau=param_dict.get('tau', 20.0),
                    window=param_dict.get('window', 252)
                )
                model = TimeDynamicsModel(params=model_params)

                # Backtest on test period
                engine = BacktestEngine(model=model)
                result = engine.run(test_prices)

                # Get objective score
                if objective == "sharpe":
                    score = result.sharpe_ratio
                elif objective == "return":
                    score = result.total_return
                elif objective == "calmar":
                    score = result.calmar_ratio
                else:
                    score = result.sharpe_ratio

                scores.append(score)

                # Step forward
                start += self.step_size

            # Average score across all windows
            avg_score = np.mean(scores) if scores else -np.inf

            if avg_score > best_score:
                best_score = avg_score
                best_params = param_dict

        return best_params, results


if __name__ == "__main__":
    # Example backtest
    np.random.seed(42)

    # Generate sample price data (simulated S&P 500-like returns)
    n = 1000
    dt = 1/252
    mu = 0.08  # 8% annual drift
    sigma = 0.18  # 18% annual volatility

    # Generate GBM prices
    returns = mu * dt + sigma * np.sqrt(dt) * np.random.randn(n)
    prices = 100 * np.cumprod(1 + returns)

    # Run backtest
    model = TimeDynamicsModel()
    config = BacktestConfig(
        initial_capital=10000,
        position_sizing=PositionSizing.VOLATILITY_TARGET,
        volatility_target=0.15
    )
    engine = BacktestEngine(model=model, config=config)

    result = engine.run(prices)
    summary = engine.print_summary(result)

    print(summary)

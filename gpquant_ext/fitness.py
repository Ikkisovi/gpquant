"""
Fitness Function: Information Sharpe + Tracking Constraints
Reward = IS(strategy vs benchmark) - penalties for violations
"""
import numpy as np
import pandas as pd


def information_sharpe(strategy_returns, benchmark_returns, ann_factor=504):
    """
    Calculate Information Sharpe (Sharpe of excess returns)

    IS = mean(excess) / std(excess) * sqrt(ann_factor)

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        ann_factor: Annualization factor (504 for half-daily data)

    Returns:
        float: Information Sharpe ratio
    """
    excess = (strategy_returns - benchmark_returns).dropna()

    if len(excess) < 10:  # Need minimum observations
        return -999.0

    mu = excess.mean()
    sigma = excess.std(ddof=1)

    if sigma < 1e-12:  # No volatility
        return 0.0 if abs(mu) < 1e-12 else (999.0 if mu > 0 else -999.0)

    IS = (mu / sigma) * np.sqrt(ann_factor)
    return IS


def relative_max_drawdown(strategy_returns, benchmark_returns):
    """
    Calculate maximum drawdown of (strategy / benchmark) relative equity

    MDD_rel = 1 - min(E_strategy / E_benchmark / max(E_strategy / E_benchmark))

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns

    Returns:
        float: Relative maximum drawdown
    """
    # Build equity curves
    E_strat = (1 + strategy_returns).cumprod()
    E_bench = (1 + benchmark_returns).cumprod()

    # Relative equity
    E_rel = (E_strat / E_bench).dropna()

    if len(E_rel) < 2:
        return 0.0

    # Drawdown
    cummax = E_rel.cummax()
    dd = (E_rel - cummax) / cummax

    mdd = abs(dd.min())
    return mdd


def absolute_excess_return(strategy_returns, benchmark_returns):
    """
    Calculate mean excess return (must be > 0 to beat benchmark)

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns

    Returns:
        float: Mean excess return
    """
    excess = (strategy_returns - benchmark_returns).dropna()
    return excess.mean()


def fitness_info_sharpe_constrained(
    strategy_returns,
    benchmark_returns,
    ann_factor=504,
    cap_mdd_rel=0.05,
    lambda_mdd=3.0,
    gamma_excess=2.0,
    delta_complexity=0.001,
    epsilon_turnover=0.1,
    formula_complexity=None,
    turnover_series=None,
    hard_constraints=False,
):
    """
    Complete fitness function with Information Sharpe + tracking constraints

    Reward = IS - penalties

    Penalties:
        1. Relative MDD > cap_mdd_rel (default 5%)
        2. Absolute excess return ≤ 0 (not beating benchmark)
        3. High formula complexity
        4. High turnover

    Args:
        strategy_returns: Series, strategy net returns
        benchmark_returns: Series, benchmark returns
        ann_factor: Annualization factor (504 for half-daily)
        cap_mdd_rel: Maximum allowed relative MDD (0.05 = 5%)
        lambda_mdd: Penalty weight for MDD violation
        gamma_excess: Penalty weight for negative excess
        delta_complexity: Penalty weight per complexity unit
        epsilon_turnover: Penalty weight for turnover
        formula_complexity: int, formula depth/nodes
        turnover_series: Series, turnover at each timestamp
        hard_constraints: If True, return -999 for any violation

    Returns:
        float: Fitness score (higher is better)
    """
    # Base reward: Information Sharpe
    IS = information_sharpe(strategy_returns, benchmark_returns, ann_factor)

    # If IS itself is degenerate, return immediately
    if IS < -900:
        return IS

    # ========================================================================
    # Constraint 1: Relative Maximum Drawdown
    # ========================================================================

    mdd_rel = relative_max_drawdown(strategy_returns, benchmark_returns)
    mdd_violation = max(0.0, mdd_rel - cap_mdd_rel)

    if hard_constraints and mdd_violation > 0:
        return -999.0

    penalty_mdd = lambda_mdd * mdd_violation

    # ========================================================================
    # Constraint 2: Absolute Excess Return > 0
    # ========================================================================

    excess_mean = absolute_excess_return(strategy_returns, benchmark_returns)

    if excess_mean <= 0:
        if hard_constraints:
            return -999.0
        penalty_excess = gamma_excess * abs(excess_mean)
    else:
        penalty_excess = 0.0

    # ========================================================================
    # Penalty 3: Formula Complexity
    # ========================================================================

    if formula_complexity is not None and formula_complexity > 0:
        penalty_complexity = delta_complexity * formula_complexity
    else:
        penalty_complexity = 0.0

    # ========================================================================
    # Penalty 4: Turnover
    # ========================================================================

    if turnover_series is not None:
        mean_turnover = turnover_series.mean()
        penalty_turnover = epsilon_turnover * mean_turnover
    else:
        penalty_turnover = 0.0

    # ========================================================================
    # Total Fitness
    # ========================================================================

    fitness = IS - penalty_mdd - penalty_excess - penalty_complexity - penalty_turnover

    return fitness


def evaluate_window_metrics(strategy_returns, benchmark_returns, ann_factor=504):
    """
    Evaluate comprehensive metrics for a single window

    Args:
        strategy_returns: Series
        benchmark_returns: Series
        ann_factor: Annualization factor

    Returns:
        dict: Metrics dictionary
    """
    # Information Sharpe
    IS = information_sharpe(strategy_returns, benchmark_returns, ann_factor)

    # Relative MDD
    mdd_rel = relative_max_drawdown(strategy_returns, benchmark_returns)

    # Excess return
    excess_mean = absolute_excess_return(strategy_returns, benchmark_returns)
    excess_cumulative = (strategy_returns - benchmark_returns).sum()

    # Strategy standalone metrics
    E_strat = (1 + strategy_returns).cumprod()
    total_return_strat = E_strat.iloc[-1] - 1

    cummax_strat = E_strat.cummax()
    dd_strat = (E_strat - cummax_strat) / cummax_strat
    mdd_strat = abs(dd_strat.min())

    sharpe_strat = strategy_returns.mean() / (strategy_returns.std() + 1e-12) * np.sqrt(ann_factor)

    # Benchmark standalone metrics
    E_bench = (1 + benchmark_returns).cumprod()
    total_return_bench = E_bench.iloc[-1] - 1

    cummax_bench = E_bench.cummax()
    dd_bench = (E_bench - cummax_bench) / cummax_bench
    mdd_bench = abs(dd_bench.min())

    sharpe_bench = benchmark_returns.mean() / (benchmark_returns.std() + 1e-12) * np.sqrt(ann_factor)

    # Constraint checks
    meets_mdd = mdd_rel <= 0.05
    meets_excess = excess_mean > 0
    meets_all = meets_mdd and meets_excess

    return {
        # Information metrics
        'information_sharpe': IS,
        'excess_mean': excess_mean,
        'excess_cumulative': excess_cumulative,

        # Relative tracking
        'relative_mdd': mdd_rel,
        'meets_mdd_constraint': meets_mdd,
        'meets_excess_constraint': meets_excess,
        'meets_all_constraints': meets_all,

        # Strategy standalone
        'strategy_return': total_return_strat,
        'strategy_mdd': mdd_strat,
        'strategy_sharpe': sharpe_strat,

        # Benchmark standalone
        'benchmark_return': total_return_bench,
        'benchmark_mdd': mdd_bench,
        'benchmark_sharpe': sharpe_bench,
    }


def print_metrics(metrics, window_name=""):
    """Pretty print metrics"""
    if window_name:
        print(f"\n{'='*80}")
        print(f"{window_name}")
        print(f"{'='*80}")

    print(f"\n📊 Information Metrics:")
    print(f"  Information Sharpe:     {metrics['information_sharpe']:>8.4f}")
    print(f"  Excess Mean (per period): {metrics['excess_mean']:>8.6f}")
    print(f"  Excess Cumulative:      {metrics['excess_cumulative']:>8.2%}")

    print(f"\n🎯 Tracking Constraints:")
    print(f"  Relative MDD:           {metrics['relative_mdd']:>8.2%}  (limit: 5%)")
    status_mdd = "✓ PASS" if metrics['meets_mdd_constraint'] else "✗ FAIL"
    print(f"  MDD Constraint:         {status_mdd:>12}")
    status_excess = "✓ PASS" if metrics['meets_excess_constraint'] else "✗ FAIL"
    print(f"  Excess Constraint:      {status_excess:>12}")
    status_all = "✓ PASS" if metrics['meets_all_constraints'] else "✗ FAIL"
    print(f"  Overall Status:         {status_all:>12}")

    print(f"\n📈 Strategy Performance:")
    print(f"  Total Return:           {metrics['strategy_return']:>8.2%}")
    print(f"  Max Drawdown:           {metrics['strategy_mdd']:>8.2%}")
    print(f"  Sharpe Ratio:           {metrics['strategy_sharpe']:>8.4f}")

    print(f"\n📊 Benchmark Performance:")
    print(f"  Total Return:           {metrics['benchmark_return']:>8.2%}")
    print(f"  Max Drawdown:           {metrics['benchmark_mdd']:>8.2%}")
    print(f"  Sharpe Ratio:           {metrics['benchmark_sharpe']:>8.4f}")

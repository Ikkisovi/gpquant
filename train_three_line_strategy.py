"""
Training script for three-line strategy on daily_am_pm_data.csv

This script:
1. Loads the AM/PM intraday data
2. Trains a genetic programming model to find optimal factors
3. Uses the three-line strategy with confirmation signals
4. Applies tracking constraints (max drawdown, return vs benchmark)
5. Evaluates performance on train/validation/test splits
"""
import pandas as pd
import numpy as np
from gpquant.SymbolicRegressor import SymbolicRegressor
from data_processor import (
    load_daily_am_pm_data,
    prepare_market_data,
    get_all_symbols,
    calculate_equal_weight_benchmark,
    check_tracking_constraints
)


def train_strategy_for_symbol(
    df: pd.DataFrame,
    symbol: str,
    benchmark_close: pd.Series,
    train_split: float = 0.6,
    val_split: float = 0.2,
    use_three_line_dual: bool = False
):
    """
    Train strategy for a single symbol

    Args:
        df: Full dataset with all symbols
        symbol: Symbol to train on
        benchmark_close: Equal-weight benchmark close prices
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        use_three_line_dual: If True, use stricter dual confirmation strategy

    Returns:
        Trained SymbolicRegressor and results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Training strategy for {symbol}")
    print(f"{'='*80}")

    # Prepare market data
    market_df = prepare_market_data(df, symbol, slippage=0.001)
    print(f"Total data points: {len(market_df)}")

    # Split data
    n_total = len(market_df)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_df = market_df.iloc[:n_train].copy()
    val_df = market_df.iloc[n_train:n_train+n_val].copy()
    test_df = market_df.iloc[n_train+n_val:].copy()

    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # Prepare benchmark for same time periods
    benchmark_train = benchmark_close.loc[train_df['dt']].values
    benchmark_val = benchmark_close.loc[val_df['dt']].values
    benchmark_test = benchmark_close.loc[test_df['dt']].values

    # Strategy selection
    strategy_name = "three_line_dual" if use_three_line_dual else "three_line"

    # Configure strategy parameters based on choice
    if use_three_line_dual:
        # Dual confirmation: stricter, needs all three MAs + factor
        transformer_kwargs = {
            "init_cash": 10000,
            "charge_ratio": 0.0002,
            "d_center": 20,
            "d_fast": 10,
            "d_slow": 40,
            "require_both": True,
            "price": train_df["C"].values,
        }
    else:
        # Single confirmation: centerline + bands + factor
        transformer_kwargs = {
            "init_cash": 10000,
            "charge_ratio": 0.0002,
            "d_center": 20,
            "d_band": 20,
            "band_width": 2.0,
            "factor_threshold": 0.5,
            "price": train_df["C"].values,
        }

    # Initialize genetic programming
    sr = SymbolicRegressor(
        population_size=500,  # Moderate population for reasonable training time
        tournament_size=20,
        generations=30,
        stopping_criteria=2.5,  # Target Sharpe ratio
        p_crossover=0.7,
        p_subtree_mutate=0.15,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(4, 7),
        init_method="half and half",
        function_set=[],  # Use all available functions
        variable_set=["O", "H", "L", "C", "V", "vwap", "mktcap", "turnover"],
        const_range=(1, 20),
        ts_const_range=(1, 30),
        build_preference=[0.75, 0.75],
        metric="sharpe ratio",
        transformer=strategy_name,
        transformer_kwargs=transformer_kwargs,
        parsimony_coefficient=0.001,
    )

    # Train
    print(f"\nTraining with {strategy_name} strategy...")
    sr.fit(train_df, train_df["C"])

    print(f"\nBest estimator: {sr.best_estimator}")
    print(f"Best fitness (Sharpe): {sr.best_fitness:.4f}")

    # Evaluate on validation set
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)

    # Update price parameter for validation
    val_kwargs = transformer_kwargs.copy()
    val_kwargs["price"] = val_df["C"].values

    sr.transformer_kwargs = val_kwargs
    val_score = sr.score(val_df, val_df["C"])
    print(f"Validation Sharpe Ratio: {val_score:.4f}")

    # Evaluate on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)

    test_kwargs = transformer_kwargs.copy()
    test_kwargs["price"] = test_df["C"].values

    sr.transformer_kwargs = test_kwargs
    test_score = sr.score(test_df, test_df["C"])
    print(f"Test Sharpe Ratio: {test_score:.4f}")

    # Get strategy asset series for constraint checking
    from gpquant.Backtester import backtester_map

    backtester = backtester_map[strategy_name]

    # Generate predictions and backtest
    train_factor = sr.best_estimator.execute(train_df)
    train_asset = backtester(
        train_df,
        train_factor,
        transformer_kwargs["init_cash"],
        transformer_kwargs["charge_ratio"],
        **{k: v for k, v in transformer_kwargs.items()
           if k not in ["init_cash", "charge_ratio"]}
    )

    val_factor = sr.best_estimator.execute(val_df)
    val_asset = backtester(
        val_df,
        val_factor,
        val_kwargs["init_cash"],
        val_kwargs["charge_ratio"],
        **{k: v for k, v in val_kwargs.items()
           if k not in ["init_cash", "charge_ratio"]}
    )

    test_factor = sr.best_estimator.execute(test_df)
    test_asset = backtester(
        test_df,
        test_factor,
        test_kwargs["init_cash"],
        test_kwargs["charge_ratio"],
        **{k: v for k, v in test_kwargs.items()
           if k not in ["init_cash", "charge_ratio"]}
    )

    # Create benchmark asset series (buy and hold with same initial cash)
    init_cash = transformer_kwargs["init_cash"]

    train_benchmark_asset = pd.Series(
        init_cash * benchmark_train / benchmark_train[0],
        index=train_df['dt']
    )
    val_benchmark_asset = pd.Series(
        init_cash * benchmark_val / benchmark_val[0],
        index=val_df['dt']
    )
    test_benchmark_asset = pd.Series(
        init_cash * benchmark_test / benchmark_test[0],
        index=test_df['dt']
    )

    # Check tracking constraints
    print("\n" + "="*80)
    print("TRACKING CONSTRAINTS CHECK")
    print("="*80)

    print("\nTrain Set:")
    train_constraints = check_tracking_constraints(
        train_asset, train_benchmark_asset, max_dd_excess=0.05
    )
    print_constraint_results(train_constraints)

    print("\nValidation Set:")
    val_constraints = check_tracking_constraints(
        val_asset, val_benchmark_asset, max_dd_excess=0.05
    )
    print_constraint_results(val_constraints)

    print("\nTest Set:")
    test_constraints = check_tracking_constraints(
        test_asset, test_benchmark_asset, max_dd_excess=0.05
    )
    print_constraint_results(test_constraints)

    return sr, {
        "train": {
            "sharpe": sr.best_fitness,
            "asset": train_asset,
            "constraints": train_constraints
        },
        "val": {
            "sharpe": val_score,
            "asset": val_asset,
            "constraints": val_constraints
        },
        "test": {
            "sharpe": test_score,
            "asset": test_asset,
            "constraints": test_constraints
        }
    }


def print_constraint_results(constraints: dict):
    """Pretty print constraint check results"""
    print(f"  Strategy Max Drawdown:   {constraints['strategy_max_drawdown']:.2%}")
    print(f"  Benchmark Max Drawdown:  {constraints['benchmark_max_drawdown']:.2%}")
    print(f"  Drawdown Excess:         {constraints['drawdown_excess']:.2%}")
    print(f"  Meets DD Constraint:     {'✓' if constraints['meets_drawdown_constraint'] else '✗'}")
    print(f"  Strategy Return:         {constraints['strategy_return']:.2%}")
    print(f"  Benchmark Return:        {constraints['benchmark_return']:.2%}")
    print(f"  Meets Return Constraint: {'✓' if constraints['meets_return_constraint'] else '✗'}")
    print(f"  Meets All Constraints:   {'✓ PASS' if constraints['meets_all_constraints'] else '✗ FAIL'}")


def main():
    """Main training loop"""
    print("="*80)
    print("THREE-LINE STRATEGY TRAINING")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = load_daily_am_pm_data("daily_am_pm_data.csv")

    # Get symbols
    symbols = get_all_symbols(df)
    print(f"Found {len(symbols)} symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")

    # Calculate equal-weight benchmark
    print("Calculating equal-weight benchmark...")
    benchmark_close = calculate_equal_weight_benchmark(df, symbols)

    # Train on first few symbols (or all if you want)
    # For demonstration, we'll train on first 3 symbols
    train_symbols = symbols[:3]  # Adjust as needed
    print(f"\nTraining on symbols: {train_symbols}")

    results = {}

    # Try both strategy versions
    for use_dual in [False, True]:
        strategy_type = "Dual Confirmation" if use_dual else "Standard"
        print(f"\n{'#'*80}")
        print(f"# Training with {strategy_type} Strategy")
        print(f"{'#'*80}")

        for symbol in train_symbols:
            try:
                sr, symbol_results = train_strategy_for_symbol(
                    df, symbol, benchmark_close,
                    train_split=0.6,
                    val_split=0.2,
                    use_three_line_dual=use_dual
                )
                results[f"{symbol}_{strategy_type}"] = {
                    "regressor": sr,
                    "results": symbol_results
                }
            except Exception as e:
                print(f"Error training {symbol}: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    for key, data in results.items():
        print(f"\n{key}:")
        print(f"  Test Sharpe: {data['results']['test']['sharpe']:.4f}")
        print(f"  Constraints: {data['results']['test']['constraints']['meets_all_constraints']}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

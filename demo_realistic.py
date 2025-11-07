"""
More realistic demo with better parameters
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


def realistic_demo():
    """Realistic training with reasonable parameters"""
    print("="*80)
    print("REALISTIC DEMO: Three-Line Strategy")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    df = load_daily_am_pm_data("daily_am_pm_data.csv")
    print(f"   Loaded {len(df)} rows")

    # Get first symbol
    symbols = get_all_symbols(df)
    symbol = symbols[0]
    print(f"   Using symbol: {symbol}")

    # Prepare market data
    print("\n2. Preparing market data...")
    market_df = prepare_market_data(df, symbol, slippage=0.001)
    print(f"   Prepared {len(market_df)} data points")

    # Split data (use more data)
    train_size = int(len(market_df) * 0.6)
    test_size = int(len(market_df) * 0.2)
    train_df = market_df.iloc[:train_size].copy()
    test_df = market_df.iloc[train_size:train_size+test_size].copy()

    print(f"   Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Calculate benchmark
    print("\n3. Calculating benchmark...")
    benchmark_close = calculate_equal_weight_benchmark(df, symbols[:10])  # Use 10 symbols
    benchmark_train = benchmark_close.loc[train_df['dt']].values
    benchmark_test = benchmark_close.loc[test_df['dt']].values

    # Setup strategy parameters
    print("\n4. Configuring three-line strategy...")
    transformer_kwargs = {
        "init_cash": 10000,
        "charge_ratio": 0.0002,
        "d_center": 15,  # Shorter for AM/PM data
        "d_band": 15,
        "band_width": 2.0,
        "factor_threshold": 0.3,  # Lower threshold
        "price": train_df["C"].values,
    }

    # Better parameters
    print("\n5. Initializing genetic programming (realistic parameters)...")
    sr = SymbolicRegressor(
        population_size=300,  # Reasonable size
        tournament_size=15,
        generations=20,  # More generations
        stopping_criteria=1.8,
        p_crossover=0.7,
        p_subtree_mutate=0.15,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(4, 6),
        init_method="half and half",
        function_set=[],
        variable_set=["O", "H", "L", "C", "V", "vwap"],
        const_range=(1, 15),
        ts_const_range=(1, 20),
        build_preference=[0.75, 0.75],
        metric="sharpe ratio",
        transformer="three_line",
        transformer_kwargs=transformer_kwargs,
        parsimony_coefficient=0.003,
    )

    # Train
    print("\n6. Training (this will take a few minutes)...")
    print("-" * 80)
    sr.fit(train_df, train_df["C"])
    print("-" * 80)

    print(f"\n   Best factor: {sr.best_estimator}")
    print(f"   Training Sharpe: {sr.best_fitness:.4f}")

    # Test
    print("\n7. Testing...")
    test_kwargs = transformer_kwargs.copy()
    test_kwargs["price"] = test_df["C"].values
    sr.transformer_kwargs = test_kwargs

    test_score = sr.score(test_df, test_df["C"])
    print(f"   Test Sharpe: {test_score:.4f}")

    # Check constraints
    print("\n8. Checking tracking constraints...")
    from gpquant.Backtester import backtester_map

    backtester = backtester_map["three_line"]

    # Get test asset
    test_factor = sr.best_estimator.execute(test_df)
    test_asset = backtester(
        test_df,
        test_factor,
        test_kwargs["init_cash"],
        test_kwargs["charge_ratio"],
        **{k: v for k, v in test_kwargs.items()
           if k not in ["init_cash", "charge_ratio"]}
    )

    # Create benchmark asset
    init_cash = test_kwargs["init_cash"]
    test_benchmark_asset = pd.Series(
        init_cash * benchmark_test / benchmark_test[0],
        index=test_df['dt']
    )

    # Check constraints
    constraints = check_tracking_constraints(
        test_asset, test_benchmark_asset, max_dd_excess=0.05
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nStrategy Performance:")
    print(f"  Sharpe Ratio:     {test_score:.4f}")
    print(f"  Max Drawdown:     {constraints['strategy_max_drawdown']:.2%}")
    print(f"  Total Return:     {constraints['strategy_return']:.2%}")

    print(f"\nBenchmark Performance:")
    print(f"  Max Drawdown:     {constraints['benchmark_max_drawdown']:.2%}")
    print(f"  Total Return:     {constraints['benchmark_return']:.2%}")

    print(f"\nConstraint Check:")
    print(f"  Drawdown Excess:  {constraints['drawdown_excess']:.2%} (limit: 5%)")
    print(f"  Return Excess:    {constraints['strategy_return'] - constraints['benchmark_return']:.2%}")
    print(f"  Status:           {'✓ PASS' if constraints['meets_all_constraints'] else '✗ FAIL'}")

    if not constraints['meets_all_constraints']:
        print(f"\n  Note: This is expected with small population/generations.")
        print(f"        For better results, use train_three_line_strategy.py")

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)


if __name__ == "__main__":
    realistic_demo()

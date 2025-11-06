"""
Tuned demo with better parameters for AM/PM intraday data
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


def tuned_demo():
    """Tuned training specifically for AM/PM intraday data"""
    print("="*80)
    print("TUNED DEMO: Three-Line Strategy for Intraday Data")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    df = load_daily_am_pm_data("daily_am_pm_data.csv")

    # Try a more volatile stock
    symbols = get_all_symbols(df)
    # Calculate volatility for each symbol to pick a more active one
    vols = []
    for sym in symbols[:10]:  # Check first 10
        sym_df = prepare_market_data(df, sym, slippage=0.001)
        vol = sym_df['C'].pct_change().std()
        vols.append((sym, vol))

    # Pick most volatile
    symbol = max(vols, key=lambda x: x[1])[0]
    print(f"   Using symbol: {symbol} (most volatile from first 10)")

    # Prepare market data
    market_df = prepare_market_data(df, symbol, slippage=0.001)
    print(f"   Prepared {len(market_df)} data points")

    # Use more data
    train_size = int(len(market_df) * 0.65)
    test_size = int(len(market_df) * 0.2)
    train_df = market_df.iloc[:train_size].copy()
    test_df = market_df.iloc[train_size:train_size+test_size].copy()

    print(f"   Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Calculate benchmark
    benchmark_close = calculate_equal_weight_benchmark(df, symbols[:15])
    benchmark_train = benchmark_close.loc[train_df['dt']].values
    benchmark_test = benchmark_close.loc[test_df['dt']].values

    # TUNED parameters for intraday data
    print("\n2. Configuring tuned three-line strategy for intraday data...")
    transformer_kwargs = {
        "init_cash": 10000,
        "charge_ratio": 0.0002,
        "d_center": 10,      # Shorter period for intraday
        "d_band": 10,        # Match center
        "band_width": 1.5,   # Tighter bands for intraday
        "factor_threshold": 0.1,  # Much lower threshold to allow more signals
        "price": train_df["C"].values,
    }

    print("\n3. Initializing genetic programming...")
    print("   Strategy: More aggressive parameters for intraday trading")
    sr = SymbolicRegressor(
        population_size=200,
        tournament_size=12,
        generations=15,
        stopping_criteria=1.0,  # Lower target, more achievable
        p_crossover=0.65,
        p_subtree_mutate=0.2,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(3, 5),
        init_method="half and half",
        function_set=[],  # Use all functions
        variable_set=["O", "H", "L", "C", "V", "vwap"],  # Core variables
        const_range=(1, 10),
        ts_const_range=(2, 15),  # Shorter windows for intraday
        build_preference=[0.7, 0.7],
        metric="sharpe ratio",
        transformer="three_line",
        transformer_kwargs=transformer_kwargs,
        parsimony_coefficient=0.01,  # Prefer simpler formulas
    )

    # Train
    print("\n4. Training...")
    print("-" * 80)
    sr.fit(train_df, train_df["C"])
    print("-" * 80)

    print(f"\n   Best factor: {sr.best_estimator}")
    print(f"   Training Sharpe: {sr.best_fitness:.4f}")

    # Test
    print("\n5. Testing...")
    test_kwargs = transformer_kwargs.copy()
    test_kwargs["price"] = test_df["C"].values
    sr.transformer_kwargs = test_kwargs

    test_score = sr.score(test_df, test_df["C"])
    print(f"   Test Sharpe: {test_score:.4f}")

    # Detailed analysis
    print("\n6. Detailed performance analysis...")
    from gpquant.Backtester import backtester_map

    backtester = backtester_map["three_line"]

    # Get test predictions and signals
    test_factor = sr.best_estimator.execute(test_df)

    # Get trading signals
    signal = backtester.f2s(
        test_factor,
        **{k: v for k, v in test_kwargs.items()
           if k not in ["init_cash", "charge_ratio"]}
    )

    n_trades = np.sum(np.abs(signal) > 0)
    print(f"   Number of trading signals: {n_trades} out of {len(signal)} periods")
    print(f"   Trading frequency: {n_trades/len(signal)*100:.1f}%")

    # Get asset series
    test_asset = backtester(
        test_df,
        test_factor,
        test_kwargs["init_cash"],
        test_kwargs["charge_ratio"],
        **{k: v for k, v in test_kwargs.items()
           if k not in ["init_cash", "charge_ratio"]}
    )

    # Benchmark
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
    print(f"  Final Asset:      ${test_asset.iloc[-1]:.2f}")

    print(f"\nBenchmark Performance:")
    print(f"  Max Drawdown:     {constraints['benchmark_max_drawdown']:.2%}")
    print(f"  Total Return:     {constraints['benchmark_return']:.2%}")
    print(f"  Final Asset:      ${test_benchmark_asset.iloc[-1]:.2f}")

    print(f"\nConstraint Check:")
    print(f"  Drawdown Excess:  {constraints['drawdown_excess']:.2%} (limit: 5%)")
    print(f"  DD Constraint:    {'✓ PASS' if constraints['meets_drawdown_constraint'] else '✗ FAIL'}")
    print(f"  Return Excess:    {constraints['strategy_return'] - constraints['benchmark_return']:.2%}")
    print(f"  Return Constraint: {'✓ PASS' if constraints['meets_return_constraint'] else '✗ FAIL'}")
    print(f"  Overall:          {'✓ PASS' if constraints['meets_all_constraints'] else '✗ FAIL'}")

    print("\n" + "="*80)
    if constraints['meets_all_constraints']:
        print("SUCCESS! Strategy meets all tracking constraints!")
    else:
        print("Strategy needs more tuning to meet constraints.")
        print("\nSuggestions:")
        if not constraints['meets_return_constraint']:
            print("  - Try longer periods for centerline/bands")
            print("  - Lower factor_threshold to generate more signals")
            print("  - Increase population_size and generations")
        if not constraints['meets_drawdown_constraint']:
            print("  - Tighten band_width to reduce risk")
            print("  - Increase factor_threshold for more conservative trades")
    print("="*80)


if __name__ == "__main__":
    tuned_demo()

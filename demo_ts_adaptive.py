"""
Quick Demo: Time-Series Adaptive Line Strategy
Tests the system with minimal configuration (fast execution)
"""
import pandas as pd
import numpy as np
from gpquant_ext.dataloader import load_panel
from gpquant_ext.features import make_features
from gpquant_ext.benchmark import equal_weight_buy_hold
from gpquant_ext.ops_ts import register_ts_ops, get_ts_whitelist
from gpquant_ext.strategy_ts import map_f_to_positions
from gpquant_ext.fitness import evaluate_window_metrics, print_metrics


def quick_demo():
    """Quick demonstration with simple factor"""
    print("="*80)
    print("QUICK DEMO: Time-Series Adaptive Line Strategy")
    print("="*80)

    # Load data
    print("\n[1/6] Loading data...")
    try:
        panel = load_panel("daily_am_pm_data.csv")
    except FileNotFoundError:
        print("✗ daily_am_pm_data.csv not found!")
        print("  Please ensure the data file is in the current directory")
        return

    # Use subset for quick demo
    symbols = panel.index.get_level_values('symbol').unique()
    demo_symbols = symbols[:5]  # First 5 symbols
    panel_demo = panel[panel.index.get_level_values('symbol').isin(demo_symbols)]

    # Use recent data only
    timestamps = panel_demo.index.get_level_values(0).unique()
    demo_timestamps = timestamps[-500:]  # Last 500 periods
    panel_demo = panel_demo[panel_demo.index.get_level_values(0).isin(demo_timestamps)]

    print(f"  Using {len(demo_symbols)} symbols, {len(demo_timestamps)} periods")

    # Register operators
    print("\n[2/6] Registering time-series operators...")
    register_ts_ops()

    # Generate features
    print("\n[3/6] Generating features...")
    X, rets, prices, meta = make_features(panel_demo)

    print(f"  Generated {len(X.columns)} features")

    # Calculate benchmark
    print("\n[4/6] Calculating benchmark (equal-weight buy-and-hold)...")
    benchmark = equal_weight_buy_hold(rets)
    print(f"  Benchmark: {len(benchmark)} periods")

    # Test with simple factor: momentum rank
    print("\n[5/6] Testing with simple factor: mom_rank_ts...")

    if 'mom_rank_ts' not in X.columns:
        print("✗ mom_rank_ts not found in features")
        return

    # Use momentum rank as the "adaptive line"
    factor = X[['mom_rank_ts']]

    # Map to positions
    w, net, metrics = map_f_to_positions(
        factor,
        prices,
        rets,
        k=1.0,
        z_L=40,
        cost_bps=0.0005,
        vol_window=60,
    )

    print(f"  Strategy executed over {len(net)} periods")
    print(f"  Mean turnover: {metrics['mean_turnover']:.4f}")

    # Evaluate
    print("\n[6/6] Evaluation...")
    results = evaluate_window_metrics(net, benchmark, ann_factor=504)

    print_metrics(results, window_name="Simple Momentum Rank Factor")

    # Summary
    print("\n" + "="*80)
    print("Demo Summary")
    print("="*80)

    print(f"\nFactor: mom_rank_ts")
    print(f"  (Robust momentum percentile in own history)")

    print(f"\nPerformance:")
    print(f"  Information Sharpe:  {results['information_sharpe']:>8.4f}")
    print(f"  Strategy Return:     {results['strategy_return']:>8.2%}")
    print(f"  Benchmark Return:    {results['benchmark_return']:>8.2%}")
    print(f"  Excess Return:       {results['excess_cumulative']:>8.2%}")

    print(f"\nRisk:")
    print(f"  Strategy MDD:        {results['strategy_mdd']:>8.2%}")
    print(f"  Benchmark MDD:       {results['benchmark_mdd']:>8.2%}")
    print(f"  Relative MDD:        {results['relative_mdd']:>8.2%}")

    print(f"\nConstraints:")
    status_mdd = "✓" if results['meets_mdd_constraint'] else "✗"
    status_excess = "✓" if results['meets_excess_constraint'] else "✗"
    status_all = "✓ PASS" if results['meets_all_constraints'] else "✗ FAIL"

    print(f"  Relative MDD ≤ 5%:    {status_mdd}")
    print(f"  Excess > 0:           {status_excess}")
    print(f"  Overall:              {status_all}")

    print(f"\nTrading:")
    print(f"  Mean Turnover:       {metrics['mean_turnover']:>8.4f}")
    print(f"  Mean Cost:           {metrics['mean_cost']:>8.6f}")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("\n1. Run full training:")
    print("   python train_ts_adaptive.py")
    print("\n2. Adjust configuration:")
    print("   Edit config_ts_adaptive.yaml")
    print("\n3. View results:")
    print("   Check results_ts_adaptive.csv")

    print("\n" + "="*80)
    print("Demo complete! System is working correctly.")
    print("="*80)


if __name__ == "__main__":
    quick_demo()

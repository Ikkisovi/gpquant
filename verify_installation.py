"""
Verification script to ensure all components are properly installed and working
"""
import sys


def verify_imports():
    """Verify all necessary imports work"""
    print("Checking imports...")
    errors = []

    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        errors.append(f"pandas: {e}")

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")

    try:
        from gpquant.Backtester import backtester_map
        print("  ✓ gpquant.Backtester")
    except ImportError as e:
        errors.append(f"gpquant.Backtester: {e}")

    try:
        from gpquant.Fitness import fitness_map
        print("  ✓ gpquant.Fitness")
    except ImportError as e:
        errors.append(f"gpquant.Fitness: {e}")

    try:
        from gpquant.SymbolicRegressor import SymbolicRegressor
        print("  ✓ gpquant.SymbolicRegressor")
    except ImportError as e:
        errors.append(f"gpquant.SymbolicRegressor: {e}")

    try:
        import data_processor
        print("  ✓ data_processor")
    except ImportError as e:
        errors.append(f"data_processor: {e}")

    return errors


def verify_strategies():
    """Verify new strategies are registered"""
    print("\nChecking strategy registration...")
    from gpquant.Backtester import backtester_map

    required_strategies = ["quantile", "three_line", "three_line_dual"]
    errors = []

    for strategy in required_strategies:
        if strategy in backtester_map:
            print(f"  ✓ {strategy}")
        else:
            errors.append(f"Strategy '{strategy}' not found in backtester_map")

    return errors


def verify_fitness_functions():
    """Verify new fitness functions are registered"""
    print("\nChecking fitness function registration...")
    from gpquant.Fitness import fitness_map

    required_fitness = [
        "annual return",
        "sharpe ratio",
        "max drawdown",
        "calmar ratio",
        "tracking constrained sharpe"
    ]
    errors = []

    for fitness in required_fitness:
        if fitness in fitness_map:
            print(f"  ✓ {fitness}")
        else:
            errors.append(f"Fitness '{fitness}' not found in fitness_map")

    return errors


def verify_data_file():
    """Verify data file exists"""
    print("\nChecking data file...")
    import os

    if os.path.exists("daily_am_pm_data.csv"):
        print("  ✓ daily_am_pm_data.csv found")
        return []
    else:
        return ["daily_am_pm_data.csv not found"]


def verify_data_loading():
    """Verify data can be loaded"""
    print("\nTesting data loading...")
    try:
        from data_processor import load_daily_am_pm_data, get_all_symbols
        df = load_daily_am_pm_data("daily_am_pm_data.csv")
        symbols = get_all_symbols(df)
        print(f"  ✓ Loaded {len(df)} rows")
        print(f"  ✓ Found {len(symbols)} symbols")
        return []
    except Exception as e:
        return [f"Data loading failed: {e}"]


def verify_strategy_signature():
    """Verify strategy functions have correct signature"""
    print("\nChecking strategy signatures...")
    from gpquant.Backtester import (
        _strategy_three_line_confirmation,
        _strategy_three_line_dual_confirmation
    )
    import inspect

    errors = []

    # Check three_line
    sig = inspect.signature(_strategy_three_line_confirmation)
    params = list(sig.parameters.keys())
    if 'factor' in params and 'price' in params:
        print("  ✓ three_line_confirmation signature")
    else:
        errors.append(f"three_line_confirmation signature incorrect: {params}")

    # Check three_line_dual
    sig = inspect.signature(_strategy_three_line_dual_confirmation)
    params = list(sig.parameters.keys())
    if 'factor' in params and 'price' in params:
        print("  ✓ three_line_dual_confirmation signature")
    else:
        errors.append(f"three_line_dual_confirmation signature incorrect: {params}")

    return errors


def main():
    """Run all verification checks"""
    print("="*80)
    print("THREE-LINE STRATEGY VERIFICATION")
    print("="*80)

    all_errors = []

    all_errors.extend(verify_imports())
    all_errors.extend(verify_strategies())
    all_errors.extend(verify_fitness_functions())
    all_errors.extend(verify_data_file())
    all_errors.extend(verify_data_loading())
    all_errors.extend(verify_strategy_signature())

    print("\n" + "="*80)
    if all_errors:
        print("VERIFICATION FAILED")
        print("="*80)
        print("\nErrors found:")
        for error in all_errors:
            print(f"  ✗ {error}")
        sys.exit(1)
    else:
        print("VERIFICATION SUCCESSFUL")
        print("="*80)
        print("\n✓ All checks passed!")
        print("\nYou can now run:")
        print("  - python demo_three_line.py          (quick demo)")
        print("  - python train_three_line_strategy.py (full training)")
        sys.exit(0)


if __name__ == "__main__":
    main()

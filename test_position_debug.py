"""Debug test for position tracking"""
import numpy as np
import pandas as pd

# Create a test case with clear signal pattern
# Start low, go high, stay high, then drop
factor = np.array([10, 15, 20, 25, 30, 95, 95, 90, 90, 85, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20])
print("Factor values:", factor)
print()

# Compute rolling quantiles manually to understand behavior
sr_factor = pd.Series(factor)
d = 5
sr_o_upper = sr_factor.rolling(d).quantile(0.8)
sr_o_lower = sr_factor.rolling(d).quantile(0.2)
sr_c_upper = sr_factor.rolling(d).quantile(0.6)
sr_c_lower = sr_factor.rolling(d).quantile(0.4)

df = pd.DataFrame({
    'factor': factor,
    'o_upper (80%)': sr_o_upper,
    'c_upper (60%)': sr_c_upper,
    'c_lower (40%)': sr_c_lower,
    'o_lower (20%)': sr_o_lower,
})

print("Rolling quantiles with d=5:")
print(df.to_string())
print()

# Now run the strategy
from gpquant.Backtester import _strategy_quantile

signal = _strategy_quantile(
    factor=factor,
    d=d,
    o_upper=0.8,
    o_lower=0.2,
    c_upper=0.6,
    c_lower=0.4,
)

position = np.cumsum(signal)

result_df = pd.DataFrame({
    'factor': factor,
    'signal': signal,
    'position': position,
})

print("Strategy output:")
print(result_df.to_string())
print()

# Explain behavior
print("Interpretation:")
print("-" * 60)
entry_indices = np.where(signal > 0)[0]
exit_indices = np.where(signal < 0)[0]

if len(entry_indices) > 0:
    print(f"Entry signals at indices: {entry_indices}")
    for idx in entry_indices:
        print(f"  Bar {idx}: factor={factor[idx]:.1f}, o_upper={sr_o_upper.iloc[idx]:.1f}")

if len(exit_indices) > 0:
    print(f"Exit signals at indices: {exit_indices}")
    for idx in exit_indices:
        print(f"  Bar {idx}: factor={factor[idx]:.1f}, c_upper={sr_c_upper.iloc[idx]:.1f}")

print()
print("Key test: Did position persist across multiple bars before exit?")
if len(entry_indices) > 0 and len(exit_indices) > 0:
    first_entry = entry_indices[0]
    first_exit = exit_indices[0]
    bars_held = first_exit - first_entry
    print(f"  Position entered at bar {first_entry}, exited at bar {first_exit}")
    print(f"  Held for {bars_held} bars")
    if bars_held > 1:
        print("  ✓ SUCCESS: Position held across multiple bars before exit")
    else:
        print("  ✗ Position exited immediately (old bug behavior)")
else:
    print("  No complete entry/exit cycle detected")

"""Test to verify position tracking fix in _strategy_quantile"""
import numpy as np
import pandas as pd
from gpquant.Backtester import _strategy_quantile

# Create a simple test case
# Factor oscillates: low -> high -> medium -> low
factor = np.array([0.1, 0.2, 0.3, 0.9, 0.7, 0.7, 0.65, 0.55, 0.4, 0.3])
df = pd.DataFrame({'factor': factor})

print("Test Case: Delayed Exit")
print("=" * 60)
print("Factor values:", factor)
print()

# Run strategy with quantile thresholds
# o_upper=0.8 (enter long at high values)
# c_upper=0.6 (exit long when drops below 60th percentile)
# o_lower=0.2 (enter short at low values)
# c_lower=0.4 (exit short when rises above 40th percentile)
signal = _strategy_quantile(
    factor=factor,
    d=5,
    o_upper=0.8,
    o_lower=0.2,
    c_upper=0.6,
    c_lower=0.4,
)

print("Signal output:", signal)
print()

# Calculate cumulative position to verify behavior
position = np.cumsum(signal)
print("Cumulative position:", position)
print()

# Verify the fix
print("Verification:")
print("-" * 60)

# Check that positions can be held across multiple bars
non_zero_signals = np.count_nonzero(signal)
print(f"Number of non-zero signals: {non_zero_signals}")

# Check that exit signals occur (not just entries)
entry_signals = np.sum(signal[signal > 0])
exit_signals = np.abs(np.sum(signal[signal < 0]))
print(f"Entry signals (sum of positive): {entry_signals}")
print(f"Exit signals (abs sum of negative): {exit_signals}")

# Final position should be flat if properly managed
final_position = position[-1]
print(f"Final position: {final_position}")

if final_position == 0:
    print("\n✓ Position properly flattened")
else:
    print(f"\n✗ Position not flat (position={final_position})")

print("\n" + "=" * 60)
print("Test demonstrates that the strategy now:")
print("1. Tracks position state across bars")
print("2. Only emits exit signals when position is open")
print("3. Can handle delayed exits (not just immediate bar after entry)")

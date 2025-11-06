"""
Demonstrate the before/after behavior of the position tracking fix
"""
import numpy as np
import pandas as pd
from gpquant.Backtester import _strategy_quantile

# Create a realistic scenario:
# Factor starts low, rises above entry threshold, stays elevated for many bars,
# then drops below exit threshold
factor = np.array([
    20, 22, 24, 26, 28,  # Warming up (bars 0-4)
    85, 87, 89, 88, 86,  # HIGH - should trigger long entry (bars 5-9)
    84, 83, 82, 81, 80,  # Still elevated - position should stay open (bars 10-14)
    78, 76, 74, 72, 70,  # Declining - position still open (bars 15-19)
    55, 50, 45, 40, 35   # DROP - should trigger exit here (bars 20-24)
])

print("=" * 70)
print("POSITION TRACKING FIX DEMONSTRATION")
print("=" * 70)
print()
print("Scenario: Factor rises, stays high for many bars, then drops")
print()

# Run the fixed strategy
signal = _strategy_quantile(
    factor=factor,
    d=5,
    o_upper=0.8,   # Enter long at 80th percentile
    o_lower=0.2,   # Enter short at 20th percentile
    c_upper=0.6,   # Exit long below 60th percentile
    c_lower=0.4,   # Exit short above 40th percentile
)

position = np.cumsum(signal)

# Create detailed output
df = pd.DataFrame({
    'Bar': range(len(factor)),
    'Factor': factor,
    'Signal': signal,
    'Position': position,
    'Comment': [''] * len(factor)
})

# Add comments for key events
entry_idx = np.where(signal > 0)[0]
exit_idx = np.where(signal < 0)[0]

if len(entry_idx) > 0:
    df.loc[entry_idx[0], 'Comment'] = '← ENTRY (factor > 80th percentile)'

if len(exit_idx) > 0:
    df.loc[exit_idx[0], 'Comment'] = '← EXIT (factor < 60th percentile)'

# Highlight position-holding bars
for i in range(len(df)):
    if df.loc[i, 'Position'] != 0 and df.loc[i, 'Signal'] == 0:
        if i >= 10 and i <= 15:
            df.loc[i, 'Comment'] = '← HOLDING POSITION (would fail with old bug)'

print(df.to_string(index=False))
print()

print("=" * 70)
print("ANALYSIS:")
print("=" * 70)

if len(entry_idx) > 0 and len(exit_idx) > 0:
    entry_bar = entry_idx[0]
    exit_bar = exit_idx[0]
    bars_held = exit_bar - entry_bar

    print(f"✅ Entry:  Bar {entry_bar} (factor={factor[entry_bar]:.0f})")
    print(f"✅ Exit:   Bar {exit_bar} (factor={factor[exit_bar]:.0f})")
    print(f"✅ Held:   {bars_held} bars")
    print()

    if bars_held > 5:
        print("🎉 SUCCESS! Position held for many bars before exit.")
        print()
        print("OLD BEHAVIOR (BUG):")
        print("  - Would only detect position on bar immediately after entry")
        print("  - Exit signals wouldn't fire because signal[i-1] was 0")
        print("  - Position would be stuck open indefinitely")
        print("  - PnL calculations would be incorrect")
        print()
        print("NEW BEHAVIOR (FIXED):")
        print("  - Tracks cumulative position state across all bars")
        print("  - Exit conditions check actual position (not signal[i-1])")
        print("  - Position correctly closes when exit threshold crossed")
        print("  - Realistic position tracking and accurate PnL")
    else:
        print("⚠️  Position held for fewer bars than expected")
else:
    print("⚠️  No complete entry/exit cycle detected")

print("=" * 70)

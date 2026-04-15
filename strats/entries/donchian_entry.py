"""Donchian Channel (price channel) entry strategy.

Backward-compatibility alias: DonchianEntryConfig/Strategy now delegate
to HLEntryConfig/Strategy from hl_entry.py. The HL channel uses raw
channel boundaries as stops (no ATR buffer) — which is the cleaned-up
successor of the original Donchian implementation.

Existing imports ``from strats.entries.donchian_entry import ...`` continue
to work unchanged.
"""

from strats.entries.hl_entry import HLEntryConfig, HLEntryStrategy  # noqa: F401

# Backward compatibility aliases
DonchianEntryConfig = HLEntryConfig
DonchianEntryStrategy = HLEntryStrategy

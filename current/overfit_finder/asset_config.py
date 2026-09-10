"""
Asset Configuration
===================
Central config for all supported assets in the multi-asset walk-forward pipeline.
Each asset has instrument-specific constants for the DSL evaluator, position sizing,
LLM prompts, and data loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class AssetConfig:
    ticker: str
    name: str
    asset_type: str            # "futures" | "equity"
    multiplier: float          # dollars per point (5.0 for MES, 1.0 for equities)
    tick_size: float
    slippage_per_side: float   # in price points
    round_trip_fee: float      # per contract/share
    sl_min: float              # minimum SL distance in points
    sl_max: float              # maximum SL distance in points
    default_sl: float          # default SL when none specified
    risk_budget: float         # dollars risked per trade
    max_contracts: int         # max position size (contracts or shares)
    overnight_session: str     # "18:00-09:30" for futures, "prev_rth" for equities
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)
    data_loader: str = "databento"  # "databento" | "polygon_csv"
    price_approx: float = 5800.0   # approximate price for seed scaling

    @property
    def instrument_details(self) -> str:
        if self.asset_type == "futures":
            return (
                f"${self.multiplier:.0f} per point, tick size {self.tick_size}, "
                f"overnight session {self.overnight_session} ET"
            )
        return (
            f"$1 per share, tick {self.tick_size}, "
            f"'overnight' levels = previous day's RTH high/low"
        )

    @property
    def overnight_details(self) -> str:
        if self.asset_type == "futures":
            return (
                f"Overnight session: {self.overnight_session} ET. "
                f"ON high/low = high/low of the overnight session before RTH open."
            )
        return (
            "No overnight futures session. 'ON high' and 'ON low' refer to the "
            "previous RTH day's high and low (09:30-16:00 ET)."
        )


# ─── Asset Configs ───────────────────────────────────────────────────────────

_MES_PRICE = 5800.0  # reference price for seed scaling


def _equity_sl(price: float, mes_sl: float) -> float:
    """Scale MES SL value to equity price."""
    return round(price / _MES_PRICE * mes_sl, 2)


MES_CONFIG = AssetConfig(
    ticker="MES",
    name="Micro E-mini S&P 500",
    asset_type="futures",
    multiplier=5.0,
    tick_size=0.25,
    slippage_per_side=0.50,
    round_trip_fee=1.24,
    sl_min=4.0,
    sl_max=32.0,
    default_sl=12.0,
    risk_budget=600.0,
    max_contracts=100,
    overnight_session="18:00-09:30",
    data_loader="databento",
    price_approx=_MES_PRICE,
)

F_CONFIG = AssetConfig(
    ticker="F",
    name="Ford Motor Company",
    asset_type="equity",
    multiplier=1.0,
    tick_size=0.01,
    slippage_per_side=0.01,
    round_trip_fee=0.0,
    sl_min=_equity_sl(10.0, 4.0),       # ~0.007 → round to 0.01
    sl_max=_equity_sl(10.0, 32.0),       # ~0.055
    default_sl=_equity_sl(10.0, 12.0),   # ~0.021
    risk_budget=600.0,
    max_contracts=5000,
    overnight_session="prev_rth",
    data_loader="polygon_csv",
    price_approx=10.0,
)

BAC_CONFIG = AssetConfig(
    ticker="BAC",
    name="Bank of America",
    asset_type="equity",
    multiplier=1.0,
    tick_size=0.01,
    slippage_per_side=0.01,
    round_trip_fee=0.0,
    sl_min=_equity_sl(45.0, 4.0),        # ~0.031
    sl_max=_equity_sl(45.0, 32.0),       # ~0.248
    default_sl=_equity_sl(45.0, 12.0),   # ~0.093
    risk_budget=600.0,
    max_contracts=2000,
    overnight_session="prev_rth",
    data_loader="polygon_csv",
    price_approx=45.0,
)

SOFI_CONFIG = AssetConfig(
    ticker="SOFI",
    name="SoFi Technologies",
    asset_type="equity",
    multiplier=1.0,
    tick_size=0.01,
    slippage_per_side=0.01,
    round_trip_fee=0.0,
    sl_min=_equity_sl(15.0, 4.0),        # ~0.010
    sl_max=_equity_sl(15.0, 32.0),       # ~0.083
    default_sl=_equity_sl(15.0, 12.0),   # ~0.031
    risk_budget=600.0,
    max_contracts=4000,
    overnight_session="prev_rth",
    data_loader="polygon_csv",
    price_approx=15.0,
)

SNAP_CONFIG = AssetConfig(
    ticker="SNAP",
    name="Snap Inc",
    asset_type="equity",
    multiplier=1.0,
    tick_size=0.01,
    slippage_per_side=0.01,
    round_trip_fee=0.0,
    sl_min=_equity_sl(12.0, 4.0),        # ~0.008
    sl_max=_equity_sl(12.0, 32.0),       # ~0.066
    default_sl=_equity_sl(12.0, 12.0),   # ~0.025
    risk_budget=600.0,
    max_contracts=5000,
    overnight_session="prev_rth",
    data_loader="polygon_csv",
    price_approx=12.0,
)

# ─── Registry ────────────────────────────────────────────────────────────────

ASSET_CONFIGS: dict[str, AssetConfig] = {
    "MES": MES_CONFIG,
    "F": F_CONFIG,
    "BAC": BAC_CONFIG,
    "SOFI": SOFI_CONFIG,
    "SNAP": SNAP_CONFIG,
}

DEFAULT_ASSET = "MES"


def get_asset_config(ticker: str) -> AssetConfig:
    """Get config for a ticker. Case-insensitive."""
    key = ticker.upper()
    if key not in ASSET_CONFIGS:
        raise ValueError(
            f"Unknown asset '{ticker}'. "
            f"Supported: {', '.join(ASSET_CONFIGS.keys())}"
        )
    return ASSET_CONFIGS[key]

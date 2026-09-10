"""
VRP Gene Definition
===================
Defines the VRPGenes dataclass and search space for the credit spread VRP system.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass


@dataclass
class VRPGenes:
    spread_type: str        # 'bull_put' | 'bear_call'
    short_strike_offset: int  # strikes OTM from ATM: [1, 2, 3, 4, 5, 7, 10]
    wing_width: int           # strikes between short and long leg: [1, 2, 3]
    tp_pct: float             # take-profit as fraction of max credit: [0.25, 0.50, 0.75]
    sl_multiple: float        # stop-loss as multiple of credit received: [2.0, 4.0]


# Run7: added bear_call spread type; expanded short_strike_offset to [1,2,3,4,5,7,10]
SEARCH_SPACE = {
    "spread_type": ["bull_put", "bear_call"],
    "short_strike_offset": [1, 2, 3, 4, 5, 7, 10],
    "wing_width": [1, 2, 3],
    "tp_pct": [0.25, 0.50, 0.75],
    "sl_multiple": [2.0, 4.0],
}


def all_genes() -> list["VRPGenes"]:
    """Return all 252 combinations in the search space (exhaustive enumeration)."""
    return [
        VRPGenes(st, sso, ww, tp, sl)
        for st, sso, ww, tp, sl in itertools.product(
            SEARCH_SPACE["spread_type"],
            SEARCH_SPACE["short_strike_offset"],
            SEARCH_SPACE["wing_width"],
            SEARCH_SPACE["tp_pct"],
            SEARCH_SPACE["sl_multiple"],
        )
    ]


def random_genes() -> VRPGenes:
    """Sample a VRPGenes uniformly at random from the search space."""
    return VRPGenes(
        spread_type=random.choice(SEARCH_SPACE["spread_type"]),
        short_strike_offset=random.choice(SEARCH_SPACE["short_strike_offset"]),
        wing_width=random.choice(SEARCH_SPACE["wing_width"]),
        tp_pct=random.choice(SEARCH_SPACE["tp_pct"]),
        sl_multiple=random.choice(SEARCH_SPACE["sl_multiple"]),
    )

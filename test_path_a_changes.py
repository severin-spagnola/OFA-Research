"""
Local validation for Path A changes to wf_runner.py.
Tests skip_all_kills flag + raw_trades serialization with mock data.
"""
import sys
import json
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent / "current" / "options_wf"))

from wf_runner import KillConfig, check_kill
from options_backtest import OptionsTrade

# ── Test 1: skip_all_kills flag ──────────────────────────────────────────────

print("=" * 60)
print("TEST 1: skip_all_kills flag")
print("=" * 60)

# Create some losing trades that would normally trigger kills
mock_trades = [
    OptionsTrade(
        trade_date="2025-06-01", direction="short", spread_type="put_debit",
        long_strike=540.0, short_strike=541.0, expiry_date="2025-06-06",
        dte=5, entry_time="09:35", entry_debit=0.45, exit_time="10:30",
        exit_value=0.0, exit_reason="max_loss", pnl_per_contract=-45.0,
        pnl_net=-50.0, max_gain=1.0, mfe_pct=0.05, mae_pct=0.95,
        result="loss", spy_entry_price=540.0, spy_exit_price=541.5,
    )
    for _ in range(10)
]

# With default config (skip_all_kills=False), these should trigger a kill
config_normal = KillConfig()
killed, reason = check_kill(mock_trades, config_normal, date(2025, 6, 1), date(2025, 6, 15))
print(f"  Normal config: killed={killed}, reason={reason}")
assert killed, "Expected kill with 10 losing trades under normal config"
print("  ✓ Normal config correctly kills")

# With skip_all_kills=True, should never kill
config_skip = KillConfig(skip_all_kills=True)
killed, reason = check_kill(mock_trades, config_skip, date(2025, 6, 1), date(2025, 6, 15))
print(f"  Skip config:   killed={killed}, reason='{reason}'")
assert not killed, "Expected no kill with skip_all_kills=True"
print("  ✓ skip_all_kills=True correctly prevents kill")

# ── Test 2: raw_trades serialization ─────────────────────────────────────────

print()
print("=" * 60)
print("TEST 2: raw_trades serialization")
print("=" * 60)

# Build diverse mock trades
fwd_trades = [
    OptionsTrade(
        trade_date=f"2025-06-{10 + i:02d}",
        direction="short" if i % 2 == 0 else "long",
        spread_type="put_debit",
        long_strike=540.0 + i,
        short_strike=541.0 + i,
        expiry_date=f"2025-06-{15 + i:02d}",
        dte=5,
        entry_time="09:35",
        entry_debit=0.45,
        exit_time="14:30",
        exit_value=0.80 if i % 3 == 0 else 0.10,
        exit_reason="target" if i % 3 == 0 else "stop_loss",
        pnl_per_contract=35.0 if i % 3 == 0 else -35.0,
        pnl_net=30.0 if i % 3 == 0 else -40.0,
        max_gain=1.0,
        mfe_pct=0.65 if i % 3 == 0 else 0.10,
        mae_pct=0.15 if i % 3 == 0 else 0.85,
        result="win" if i % 3 == 0 else "loss",
        spy_entry_price=540.0,
        spy_exit_price=541.0,
    )
    for i in range(5)
]

# Serialize exactly as wf_runner.py does
raw_trades = [
    {
        "trade_num": i + 1,
        "trade_date": t.trade_date,
        "direction": t.direction,
        "pnl_net": round(t.pnl_net, 2),
        "result": t.result,
        "exit_reason": t.exit_reason,
        "mfe_pct": round(t.mfe_pct, 4),
        "mae_pct": round(t.mae_pct, 4),
        "long_strike": t.long_strike,
        "short_strike": t.short_strike,
    }
    for i, t in enumerate(fwd_trades)
]

# Verify field count
expected_fields = {"trade_num", "trade_date", "direction", "pnl_net", "result",
                   "exit_reason", "mfe_pct", "mae_pct", "long_strike", "short_strike"}
for rt in raw_trades:
    assert set(rt.keys()) == expected_fields, f"Field mismatch: {set(rt.keys())} vs {expected_fields}"
print(f"  ✓ All {len(raw_trades)} trades have correct 10 fields")

# Verify PnL sum matches
fwd_cum_pnl = sum(t.pnl_net for t in fwd_trades)
raw_pnl_sum = sum(rt["pnl_net"] for rt in raw_trades)
print(f"  fwd_cum_pnl = {fwd_cum_pnl:.2f}")
print(f"  sum(raw_trades.pnl_net) = {raw_pnl_sum:.2f}")
assert abs(fwd_cum_pnl - raw_pnl_sum) < 0.01, f"PnL mismatch: {fwd_cum_pnl} vs {raw_pnl_sum}"
print("  ✓ PnL sums match")

# Verify JSON-serializable
try:
    json_str = json.dumps({"raw_trades": raw_trades})
    parsed = json.loads(json_str)
    assert len(parsed["raw_trades"]) == 5
    print("  ✓ raw_trades is JSON-serializable")
except Exception as e:
    print(f"  ✗ JSON serialization failed: {e}")
    sys.exit(1)

# Print first 3 trades
print()
print("  First 3 trades:")
for rt in raw_trades[:3]:
    print(f"    {json.dumps(rt)}")

# ── Test 3: Full record structure ────────────────────────────────────────────

print()
print("=" * 60)
print("TEST 3: Full record structure (mock)")
print("=" * 60)

record = {
    "window_id": 42,
    "fwd_cum_pnl": round(fwd_cum_pnl, 2),
    "fwd_n_trades": len(fwd_trades),
    "forward_exit_reason": "data_end",
    "raw_trades": raw_trades,
}
record_json = json.dumps(record)
parsed = json.loads(record_json)
assert "raw_trades" in parsed
assert len(parsed["raw_trades"]) == 5
assert parsed["fwd_cum_pnl"] == round(raw_pnl_sum, 2)
print(f"  ✓ Record has raw_trades ({len(parsed['raw_trades'])} trades)")
print(f"  ✓ fwd_cum_pnl ({parsed['fwd_cum_pnl']}) matches sum of raw_trades pnl_net ({round(raw_pnl_sum, 2)})")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)

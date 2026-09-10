"""
Simple HTTP server for the OverfitAlpha options dashboard.
Serves HTML at GET / and mock API data at GET /api/options/live.
No dependencies beyond stdlib.

Usage:
    python serve.py
"""

import http.server
import json
import random
import math
from datetime import datetime, timedelta

PORT = 8050


# ── Mock Data Generator ──────────────────────────────────────────────────────

def _generate_mock_data():
    """Generate realistic mock data for the options livesim dashboard."""
    random.seed(42)

    today = datetime(2026, 3, 14)
    sim_start = today - timedelta(days=60)

    # ── Kill Config ──
    kill_config = {
        "max_dd_dollars": 3000,
        "max_consec_losses": 5,
        "stl_max_loss": 800,
        "stl_window_days": 5,
        "max_days_alive": 90,
        "min_trades_for_eval": 8,
        "min_wr_after_eval": 0.35,
        "max_losing_streak_days": 10,
    }

    # ── Risk Config ──
    risk_config = {
        "max_portfolio_dd": 5000,
        "scale_threshold": 1.5,
        "max_contracts": 5,
        "base_contract_dd": 3000,
        "stl_window": 5,
        "daily_loss_limit": 1500,
    }

    # ── Live Score Config ──
    live_score_config = {
        "base_score": 1.0,
        "wr_bonus": 0.3,
        "wr_threshold": 0.5,
        "pnl_bonus": 0.2,
        "pnl_norm": 500,
        "streak_bonus": 0.1,
        "dd_penalty": 0.5,
    }

    # ── 5 Active Strategies ──
    strategies = [
        {
            "strategy_id": "SPY_breakout_L_r47",
            "status": "active",
            "start_date": "2026-01-28",
            "n_trades": 18,
            "cum_pnl": 1247.50,
            "current_dd": 380.00,
            "contracts": 2,
            "live_score": 1.72,
            "genes": {
                "archetype": "breakout",
                "direction": "long",
                "trade_type": "call_debit_spread",
                "time_window": "09:35-11:30",
                "entry_conditions": [
                    "price > VWAP",
                    "RSI_14 > 55",
                    "volume_ratio > 1.5",
                    "ATR_pct > 0.8%",
                ],
                "exit_rules": [
                    "take_profit: 80% max_gain",
                    "stop_loss: -40% premium",
                    "time_stop: 14:30",
                ],
            },
            "training_stats": {
                "fitness": 2.847,
                "pnl": 4280.00,
                "wr": 0.62,
                "sharpe": 1.41,
            },
            "forward_stats": {
                "trades": 18,
                "pnl": 1247.50,
                "wr": 0.61,
                "dd": 380.00,
                "days_alive": 32,
                "peak_equity": 1627.50,
                "consec_losses": 1,
                "stl_loss": 120.00,
            },
        },
        {
            "strategy_id": "QQQ_meanrev_S_r51",
            "status": "active",
            "start_date": "2026-02-05",
            "n_trades": 14,
            "cum_pnl": 830.00,
            "current_dd": 210.00,
            "contracts": 1,
            "live_score": 1.45,
            "genes": {
                "archetype": "mean_revert",
                "direction": "short",
                "trade_type": "put_debit_spread",
                "time_window": "10:00-14:00",
                "entry_conditions": [
                    "price > upper_BB_20",
                    "RSI_14 > 72",
                    "MACD_hist declining",
                    "VIX > 18",
                ],
                "exit_rules": [
                    "take_profit: 60% max_gain",
                    "stop_loss: -35% premium",
                    "revert_to_mean: price < SMA_20",
                ],
            },
            "training_stats": {
                "fitness": 2.412,
                "pnl": 3150.00,
                "wr": 0.58,
                "sharpe": 1.22,
            },
            "forward_stats": {
                "trades": 14,
                "pnl": 830.00,
                "wr": 0.57,
                "dd": 210.00,
                "days_alive": 26,
                "peak_equity": 1040.00,
                "consec_losses": 0,
                "stl_loss": 0.00,
            },
        },
        {
            "strategy_id": "SPY_momentum_L_r53",
            "status": "active",
            "start_date": "2026-02-12",
            "n_trades": 11,
            "cum_pnl": -180.00,
            "current_dd": 640.00,
            "contracts": 1,
            "live_score": 0.88,
            "genes": {
                "archetype": "momentum",
                "direction": "long",
                "trade_type": "long_call",
                "time_window": "09:45-13:00",
                "entry_conditions": [
                    "EMA_9 > EMA_21",
                    "ADX_14 > 25",
                    "close > yesterday_high",
                    "OBV trending up",
                ],
                "exit_rules": [
                    "take_profit: 100% premium",
                    "stop_loss: -50% premium",
                    "trailing_stop: 30%",
                ],
            },
            "training_stats": {
                "fitness": 1.953,
                "pnl": 2870.00,
                "wr": 0.54,
                "sharpe": 0.98,
            },
            "forward_stats": {
                "trades": 11,
                "pnl": -180.00,
                "wr": 0.45,
                "dd": 640.00,
                "days_alive": 21,
                "peak_equity": 460.00,
                "consec_losses": 2,
                "stl_loss": 340.00,
            },
        },
        {
            "strategy_id": "IWM_breakout_S_r48",
            "status": "active",
            "start_date": "2026-02-18",
            "n_trades": 9,
            "cum_pnl": 415.00,
            "current_dd": 150.00,
            "contracts": 1,
            "live_score": 1.31,
            "genes": {
                "archetype": "breakout",
                "direction": "short",
                "trade_type": "put_vertical",
                "time_window": "09:35-12:00",
                "entry_conditions": [
                    "price < prior_day_low",
                    "gap_down > 0.3%",
                    "RVOL > 1.8",
                    "sector_breadth < -0.5",
                ],
                "exit_rules": [
                    "take_profit: 70% max_gain",
                    "stop_loss: -45% premium",
                    "time_stop: 13:00",
                ],
            },
            "training_stats": {
                "fitness": 2.156,
                "pnl": 3410.00,
                "wr": 0.56,
                "sharpe": 1.15,
            },
            "forward_stats": {
                "trades": 9,
                "pnl": 415.00,
                "wr": 0.56,
                "dd": 150.00,
                "days_alive": 17,
                "peak_equity": 565.00,
                "consec_losses": 1,
                "stl_loss": 85.00,
            },
        },
        {
            "strategy_id": "SPY_fade_S_r55",
            "status": "active",
            "start_date": "2026-03-01",
            "n_trades": 6,
            "cum_pnl": 290.00,
            "current_dd": 0.00,
            "contracts": 1,
            "live_score": 1.18,
            "genes": {
                "archetype": "fade",
                "direction": "short",
                "trade_type": "put_debit_spread",
                "time_window": "09:35-10:30",
                "entry_conditions": [
                    "gap_up > 0.5%",
                    "pre_market_volume < avg",
                    "VIX_term_structure inverted",
                    "prior_day_range > 1.2 * ATR",
                ],
                "exit_rules": [
                    "take_profit: gap_fill",
                    "stop_loss: -30% premium",
                    "time_stop: 11:00",
                ],
            },
            "training_stats": {
                "fitness": 2.034,
                "pnl": 2640.00,
                "wr": 0.60,
                "sharpe": 1.08,
            },
            "forward_stats": {
                "trades": 6,
                "pnl": 290.00,
                "wr": 0.67,
                "dd": 0.00,
                "days_alive": 10,
                "peak_equity": 290.00,
                "consec_losses": 0,
                "stl_loss": 0.00,
            },
        },
    ]

    # ── 3 Completed Regimes ──
    completed_regimes = [
        {
            "regime_id": "r39_SPY_trend_L",
            "train_period": "2025-06-01 to 2025-09-30",
            "forward_period": "2025-10-15 to 2025-12-20",
            "days_alive": 47,
            "trades": 28,
            "pnl": 2180.00,
            "wr": 0.61,
            "max_dd": 720.00,
            "max_contracts": 3,
            "death_reason": "max_days_alive",
        },
        {
            "regime_id": "r41_QQQ_scalp_S",
            "train_period": "2025-07-01 to 2025-10-31",
            "forward_period": "2025-11-10 to 2026-01-15",
            "days_alive": 42,
            "trades": 35,
            "pnl": 1450.00,
            "wr": 0.57,
            "max_dd": 890.00,
            "max_contracts": 2,
            "death_reason": "max_days_alive",
        },
        {
            "regime_id": "r44_IWM_fade_L",
            "train_period": "2025-08-01 to 2025-11-30",
            "forward_period": "2025-12-10 to 2026-01-22",
            "days_alive": 28,
            "trades": 16,
            "pnl": -1240.00,
            "wr": 0.38,
            "max_dd": 1640.00,
            "max_contracts": 1,
            "death_reason": "max_dd_dollars",
        },
    ]

    # ── 30 Recent Trades ──
    trade_archetypes = [
        ("SPY_breakout_L_r47", "long", "call_debit_spread"),
        ("QQQ_meanrev_S_r51", "short", "put_debit_spread"),
        ("SPY_momentum_L_r53", "long", "long_call"),
        ("IWM_breakout_S_r48", "short", "put_vertical"),
        ("SPY_fade_S_r55", "short", "put_debit_spread"),
    ]
    exit_reasons = [
        "take_profit", "take_profit", "take_profit",
        "stop_loss", "stop_loss",
        "time_stop", "trailing_stop", "revert_to_mean",
    ]

    trades = []
    cum_pnl = 0.0
    for i in range(30):
        day_offset = 60 - i * 2
        trade_date = (today - timedelta(days=max(day_offset, 0))).strftime("%Y-%m-%d")
        strat_idx = i % 5
        sid, direction, ttype = trade_archetypes[strat_idx]
        reason = random.choice(exit_reasons)

        # Generate realistic option PnL
        if reason == "take_profit":
            pnl_per_ct = round(random.uniform(80, 580), 2)
        elif reason == "stop_loss":
            pnl_per_ct = round(random.uniform(-400, -100), 2)
        elif reason == "trailing_stop":
            pnl_per_ct = round(random.uniform(30, 250), 2)
        else:
            pnl_per_ct = round(random.uniform(-200, 300), 2)

        contracts = random.choice([1, 1, 1, 2, 2])
        scaled_pnl = round(pnl_per_ct * contracts, 2)
        cum_pnl += scaled_pnl

        entry_hour = random.randint(9, 12)
        entry_min = random.randint(30, 59) if entry_hour == 9 else random.randint(0, 59)
        exit_hour = min(entry_hour + random.randint(0, 3), 15)
        exit_min = random.randint(0, 59)

        trades.append({
            "date": trade_date,
            "strategy_id": sid,
            "direction": direction,
            "trade_type": ttype,
            "entry_time": f"{entry_hour:02d}:{entry_min:02d}",
            "exit_time": f"{exit_hour:02d}:{exit_min:02d}",
            "exit_reason": reason,
            "pnl_per_ct": pnl_per_ct,
            "contracts": contracts,
            "scaled_pnl": scaled_pnl,
            "cum_pnl": round(cum_pnl, 2),
        })

    # ── Equity Curve ──
    equity_curve = []
    daily_pnl_map = {}
    for t in trades:
        if t["date"] not in daily_pnl_map:
            daily_pnl_map[t["date"]] = 0
        daily_pnl_map[t["date"]] += t["scaled_pnl"]

    running = 0.0
    n_trades_running = 0
    for date_str in sorted(daily_pnl_map.keys()):
        running += daily_pnl_map[date_str]
        day_trades = [t for t in trades if t["date"] == date_str]
        n_trades_running += len(day_trades)
        equity_curve.append({
            "date": date_str,
            "cum_pnl": round(running, 2),
            "n_trades": len(day_trades),
        })

    # Fill in missing days with flat equity
    if equity_curve:
        filled = []
        start = datetime.strptime(equity_curve[0]["date"], "%Y-%m-%d")
        end = datetime.strptime(equity_curve[-1]["date"], "%Y-%m-%d")
        eq_map = {e["date"]: e for e in equity_curve}
        last_val = 0
        d = start
        while d <= end:
            ds = d.strftime("%Y-%m-%d")
            if ds in eq_map:
                last_val = eq_map[ds]["cum_pnl"]
                filled.append(eq_map[ds])
            else:
                filled.append({"date": ds, "cum_pnl": last_val, "n_trades": 0})
            d += timedelta(days=1)
        equity_curve = filled

    # ── Data Calendar ──
    data_calendar = []
    for i in range(42):
        d = (today - timedelta(days=41 - i))
        ds = d.strftime("%Y-%m-%d")
        is_weekday = d.weekday() < 5
        day_trades = [t for t in trades if t["date"] == ds]
        day_pnl = sum(t["scaled_pnl"] for t in day_trades) if day_trades else None
        data_calendar.append({
            "date": ds,
            "has_data": is_weekday and len(day_trades) > 0,
            "pnl": round(day_pnl, 2) if day_pnl is not None else None,
        })

    # ── Build Info ──
    build_info = {
        "git_sha": "a3f7c2e",
        "branch": "main",
        "build_time": "2026-03-14T08:30:00Z",
        "cache_bust": "1710408600",
        "python_version": "3.11.7",
        "server": "local",
    }

    # ── Session ──
    session = {
        "mode": "livesim",
        "date": today.strftime("%Y-%m-%d"),
        "peak_concurrent": 5,
    }

    return {
        "session": session,
        "strategies": strategies,
        "completed_regimes": completed_regimes,
        "trades": trades,
        "equity_curve": equity_curve,
        "kill_config": kill_config,
        "risk_config": risk_config,
        "live_score_config": live_score_config,
        "build_info": build_info,
        "data_calendar": data_calendar,
    }


# ── Pre-generate mock data (once) ────────────────────────────────────────────
MOCK_DATA = _generate_mock_data()
MOCK_JSON = json.dumps(MOCK_DATA)


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self._serve_dashboard()
        elif self.path == "/api/options/live":
            self._serve_api()
        else:
            self.send_error(404, "Not Found")

    def _serve_dashboard(self):
        from options_dashboard import options_dashboard_html
        html = options_dashboard_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_api(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(MOCK_JSON.encode("utf-8"))

    def log_message(self, format, *args):
        # Quieter logging — only show requests
        if args and len(args) >= 3:
            method_path = args[0] if args else ""
            status = args[1] if len(args) > 1 else ""
            print(f"  {method_path} -> {status}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # Ensure the dashboard module is importable
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    server = http.server.HTTPServer(("localhost", PORT), DashboardHandler)
    print(f"Options Dashboard running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()

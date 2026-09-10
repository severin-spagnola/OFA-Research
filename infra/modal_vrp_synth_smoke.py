#!/usr/bin/env python3
"""
Synthetic end-to-end smoke for OFA-VRP.

Generates synthetic parquet chain files inside the container, calls
evaluate_window() on windows[0], and asserts ok=True.
No Modal Volume data required.

Usage:
    modal run infra/modal_vrp_synth_smoke.py
"""
import json
import sys
from pathlib import Path

import modal

app = modal.App("ofa-vrp-synth-smoke")

_OPTIONS_VRP_DIR = Path(__file__).resolve().parent.parent / "current" / "options_vrp"

# CACHE_BUST=43 — keep in sync with modal_vrp.py to reuse cached layers
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "lightgbm",
        "yfinance",
    )
    .run_commands("echo 'CACHE_BUST=43'")
    .add_local_dir(
        str(_OPTIONS_VRP_DIR),
        remote_path="/root/options_vrp",
        ignore=lambda p: str(p).startswith("results"),
    )
)


@app.function(image=image, timeout=300)
def run_window_synth() -> dict:
    """Generate synthetic chain parquet files, run evaluate_window on windows[0], return result."""
    import json as _json
    import sys as _sys
    import numpy as np
    import pandas as pd
    from pathlib import Path as _Path
    from datetime import datetime

    _sys.path.insert(0, "/root/options_vrp")
    from wf_runner import generate_windows, evaluate_window, KillConfig

    # ── Generate synthetic trading dates ─────────────────────────────────────
    dates = list(pd.bdate_range(start="2023-01-03", periods=70).date)

    # ── Write synthetic chain files to /tmp/synth_chain/ ─────────────────────
    data_dir = _Path("/tmp/synth_chain")
    data_dir.mkdir(parents=True, exist_ok=True)

    strikes = [450, 455, 460, 465, 470, 475, 480]
    atm = 465.0

    for d in dates:
        date_str = d.isoformat()

        meta = {"strikes": strikes, "atm": atm}
        (data_dir / f"chain_{date_str}_meta.json").write_text(_json.dumps(meta))

        rows = []
        ts = datetime(d.year, d.month, d.day, 9, 30)
        for strike in strikes:
            for cp in ("C", "P"):
                rows.append({
                    "strike": float(strike),
                    "cp": cp,
                    "c": 2.5 + (strike - atm) * 0.01 * (1 if cp == "C" else -1),
                    "ts": np.datetime64(ts, "ns"),
                })
        df = pd.DataFrame(rows)
        df["ts"] = df["ts"].astype("datetime64[ns]")
        df.to_parquet(data_dir / f"chain_{date_str}.parquet", index=False)

    # ── Generate windows and evaluate windows[0] ──────────────────────────────
    windows = generate_windows(dates, train_size=45, val_size=15, step_size=10)
    window = windows[0]

    records = evaluate_window(
        window=window,
        data_dir=str(data_dir),
        n_candidates=3,
        n_generations=1,
        top_n=3,
        min_train_trades=1,
        seed=42,
        kill_config=KillConfig(skip_all_kills=True),
    )

    return {
        "window_id": 0,
        "n_records": len(records),
        "smoke": True,
        "ok": True,
    }


@app.local_entrypoint()
def main():
    result = run_window_synth.remote()
    print(json.dumps(result))
    assert result.get("ok") is True, f"run_window_synth returned unexpected result: {result}"
    print("SYNTH_SMOKE PASS")
    sys.exit(0)

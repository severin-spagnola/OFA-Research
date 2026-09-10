#!/usr/bin/env python3
"""
Modal app for OFA-VRP walk-forward search.

Usage:
    modal run infra/modal_vrp.py -- --data-dir /path/to/options_5dte
    modal run infra/modal_vrp.py -- --data-dir /path/to/options_5dte --candidates 400 --seed 456
    modal run infra/modal_vrp.py -- --data-dir /path/to/options_5dte --out results/options_vrp/wf_results_modal.jsonl

Data access:
    Remote workers read chain files from a Modal Volume named "ofa-vrp-data".
    Upload your options_5dte directory to that volume before dispatching:
        modal volume put ofa-vrp-data /path/to/options_5dte options_5dte
    Then pass --remote-data-dir /root/data/options_5dte (the default).
"""
import json
import sys
from datetime import date
from pathlib import Path

import modal

app = modal.App("ofa-vrp-wf")

_OPTIONS_VRP_DIR = Path(__file__).resolve().parent.parent / "current" / "options_vrp"

# CACHE_BUST=43 — force image rebuild to pick up latest wf_runner.py
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
        # exclude results/ to avoid bundling large JSONL outputs into the image
        ignore=lambda p: str(p).startswith("results"),
    )
)

data_vol = modal.Volume.from_name("ofa-vrp-data", create_if_missing=True)


# ─── Serialization helpers ────────────────────────────────────────────────────

def _window_to_dict(window) -> dict:
    """Serialize a WFWindow to a JSON-safe dict."""
    return {
        "window_id": window.window_id,
        "train_start": window.train_start.isoformat(),
        "train_end": window.train_end.isoformat(),
        "val_start": window.val_start.isoformat(),
        "val_end": window.val_end.isoformat(),
        "train_days": [d.isoformat() for d in window.train_days],
        "val_days": [d.isoformat() for d in window.val_days],
        "fwd_days": [d.isoformat() for d in window.fwd_days],
    }


def _dict_to_window(d: dict):
    """Deserialize a window dict into a WFWindow. Must run inside the container."""
    from wf_runner import WFWindow  # available after sys.path.insert in run_window
    return WFWindow(
        window_id=d["window_id"],
        train_start=date.fromisoformat(d["train_start"]),
        train_end=date.fromisoformat(d["train_end"]),
        val_start=date.fromisoformat(d["val_start"]),
        val_end=date.fromisoformat(d["val_end"]),
        train_days=[date.fromisoformat(s) for s in d["train_days"]],
        val_days=[date.fromisoformat(s) for s in d["val_days"]],
        fwd_days=[date.fromisoformat(s) for s in d["fwd_days"]],
    )


# ─── Smoke sentinel ───────────────────────────────────────────────────────────

@app.function(image=image, timeout=120)
def noop_window() -> dict:
    """Validate container imports; return a smoke sentinel dict."""
    sys.path.insert(0, "/root/options_vrp")
    from wf_runner import evaluate_window, WFWindow, KillConfig  # noqa: F401
    return {"window_id": "smoke_sentinel", "smoke": True}


def smoke():
    sentinel = noop_window.remote()
    print(json.dumps(sentinel))
    _repo_root = Path(__file__).resolve().parent.parent
    out_path = _repo_root / "current" / "options_vrp" / "results" / "options_vrp" / "regime_db.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sentinel) + "\n")
    print("SMOKE PASS")


# ─── Synthetic smoke path ─────────────────────────────────────────────────────

@app.function(image=image, timeout=300)
def run_window_synth() -> list[dict]:
    """Generate synthetic chain parquet files, run evaluate_window on windows[0], return result."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path as _Path
    from datetime import datetime

    sys.path.insert(0, "/root/options_vrp")
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

    return records



def smoke_synth():
    records: list[dict] = run_window_synth.remote()
    _repo_root = Path(__file__).resolve().parent.parent
    out_path = _repo_root / "current" / "options_vrp" / "results" / "options_vrp" / "smoke_test.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[smoke_synth] {len(records)} records written to {out_path}")


# ─── Remote function ──────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={"/root/data": data_vol},
    cpu=4,
    timeout=3600,
)
def run_window(payload: dict) -> list[dict]:
    """Deserialize a window dict into WFWindow, call evaluate_window(), return regime records."""
    sys.path.insert(0, "/root/options_vrp")
    from wf_runner import evaluate_window, KillConfig

    window = _dict_to_window(payload["window"])
    kc_overrides = payload.get("kill_config_overrides", {})
    kill_config = KillConfig(**kc_overrides) if kc_overrides else KillConfig()
    records = evaluate_window(
        window=window,
        data_dir=payload["data_dir"],
        n_candidates=payload.get("n_candidates", 100),
        n_generations=payload.get("n_generations", 1),
        top_n=payload.get("top_n", 20),
        min_train_trades=payload.get("min_train_trades", 5),
        seed=payload.get("seed", 42),
        kill_config=kill_config,
        verbose=payload.get("verbose", False),
    )
    print(f"[run_window] window_id={window.window_id} records={len(records)}", flush=True)
    return records


@app.function(
    image=image,
    volumes={"/root/data": data_vol},
    timeout=120,
)
def get_volume_days(remote_data_dir: str) -> list[str]:
    """Glob chain_*_meta.json + matching .parquet files; return sorted ISO date strings."""
    from pathlib import Path as _Path
    data_path = _Path(remote_data_dir)
    dates = []
    for meta_file in sorted(data_path.glob("chain_*_meta.json")):
        stem = meta_file.stem  # e.g. "chain_2024-01-02_meta"
        date_str = stem[len("chain_"):-len("_meta")]
        if (data_path / f"chain_{date_str}.parquet").exists():
            dates.append(date_str)
    return sorted(dates)


# ─── Payload builder ──────────────────────────────────────────────────────────

def _make_payloads(
    remote_data_dir: str,
    train_days: int,
    val_days: int,
    step_days: int,
    candidates: int,
    generations: int,
    top_n: int,
    min_train_trades: int,
    seed: int,
    max_windows: int,
    smoke: bool,
    log_fn=print,
) -> list[dict]:
    """Discover windows and build run_window payloads. Does not filter done_ids."""
    _repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_repo_root / "current" / "options_vrp"))

    from wf_runner import generate_windows
    log_fn(f"[_make_payloads] discovering days from volume: {remote_data_dir}")
    iso_dates = get_volume_days.remote(remote_data_dir)
    if not iso_dates:
        log_fn(f"[_make_payloads] ERROR: no available days found in volume at {remote_data_dir}")
        return []
    all_days = [date.fromisoformat(s) for s in iso_dates]
    windows = generate_windows(
        all_days,
        train_size=train_days,
        val_size=val_days,
        step_size=step_days,
    )
    log_fn(
        f"[_make_payloads] {len(windows)} windows generated from volume | "
        f"train={train_days}d val={val_days}d step={step_days}d | "
        # NOTE: 'days' is option-chain parquet count from volume, NOT SPY daily rows
        f"days={len(all_days)}"
    )

    if smoke:
        windows = windows[:2]
        log_fn(f"[_make_payloads] --smoke: capped at {len(windows)} window(s) from real volume data")
    elif max_windows:
        windows = windows[:max_windows]
        log_fn(f"[_make_payloads] --max-windows={max_windows}: truncated to {len(windows)} window(s)")

    payloads = [
        {
            "window":                _window_to_dict(w),
            "data_dir":              remote_data_dir,
            "n_candidates":          candidates,
            "n_generations":         generations,
            "top_n":                 top_n,
            "min_train_trades":      min_train_trades,
            "seed":                  seed + w.window_id * 137,
            "verbose":               False,
            "kill_config_overrides": {"skip_all_kills": True},
        }
        for w in windows
    ]
    return payloads


# ─── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def dispatch(
    data_dir: str = "",
    remote_data_dir: str = "/root/data/options_5dte",
    train_days: int = 45,
    val_days: int = 15,
    step_days: int = 10,
    candidates: int = 500,
    generations: int = 1,
    top_n: int = 20,
    min_train_trades: int = 1,
    seed: int = 42,
    out: str = "current/options_vrp/results/options_vrp/regime_db.jsonl",
    max_windows: int = 0,
    smoke: bool = False,
):
    # ── Set up persistent log tee to /tmp/modal_run.log ──────────────────────
    _log_path = Path("/tmp/modal_run.log")
    _log_fh = open(_log_path, "w")  # truncate at start of each run

    def _log(msg: str, flush: bool = True) -> None:
        _log_fh.write(msg + "\n")
        if flush:
            _log_fh.flush()

    if smoke:
        out = "current/options_vrp/results/options_vrp/smoke_test.jsonl"

    # ── Set up output path early so sentinels can be written on any exit ──────
    # Anchor relative paths to repo root so the correct file is written
    # regardless of the CWD when `modal run` is invoked.
    _repo_root = Path(__file__).resolve().parent.parent
    out_path = Path(out) if Path(out).is_absolute() else _repo_root / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if smoke:
        # Smoke runs always start fresh: truncate so done_ids is empty and
        # the run always actually dispatches windows to Modal.
        out_path.write_text("")
        # Write a "started" sentinel immediately — guarantees ≥1 line even if
        # the process crashes before any other write succeeds.
        with open(out_path, "a") as _start_fh:
            _start_fh.write(json.dumps({
                "window_id": "smoke_started",
                "smoke": True,
                "note": "dispatch() reached file-I/O checkpoint",
            }) + "\n")
    else:
        # Always touch the output file so its existence proves dispatch() ran,
        # even when 0 records are written.
        out_path.touch(exist_ok=True)

    _log("[dispatch] modal_vrp.py started")
    print("[dispatch] modal_vrp.py started", flush=True)

    # ── Build payloads (window discovery + payload construction) ──────────────
    try:
        payloads = _make_payloads(
            remote_data_dir=remote_data_dir,
            train_days=train_days,
            val_days=val_days,
            step_days=step_days,
            candidates=candidates,
            generations=generations,
            top_n=top_n,
            min_train_trades=min_train_trades,
            seed=seed,
            max_windows=max_windows,
            smoke=smoke,
            log_fn=_log,
        )
    except BaseException as _make_exc:
        import traceback as _tb
        err_msg = f"[dispatch] ERROR in _make_payloads: {type(_make_exc).__name__}: {_make_exc}"
        _log(err_msg)
        _log(_tb.format_exc())
        print(err_msg, flush=True)
        print(_tb.format_exc(), flush=True)
        try:
            _log_fh.close()
        except Exception:
            pass
        if smoke:
            try:
                with open(out_path, "a") as _ef:
                    _ef.write(json.dumps({
                        "window_id": "smoke_error",
                        "smoke": True,
                        "error": "make_payloads_failed",
                        "exc": str(_make_exc),
                        "exc_type": type(_make_exc).__name__,
                    }) + "\n")
            except Exception as _write_exc:
                print(f"[dispatch] WARN: could not write smoke_error sentinel: {_write_exc}", flush=True)
        sys.exit(1)

    if not payloads:
        err_msg = "[dispatch] ERROR: no payloads generated — check volume data"
        print(err_msg, flush=True)
        _log(err_msg)
        if smoke:
            _smoke_sent = {
                "window_id": "smoke_sentinel",
                "smoke": True,
                "error": "no_payloads",
                "note": "smoke run failed: _make_payloads returned empty list (volume empty?)",
            }
            with open(out_path, "a") as _sf:
                _sf.write(json.dumps(_smoke_sent) + "\n")
            print(f"[dispatch] smoke sentinel written to {out_path}", flush=True)
        return

    # ── Dedupe against existing output JSONL ──────────────────────────────────

    done_ids: set[int] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    wid = rec.get("window_id")
                    if wid is not None:
                        try:
                            done_ids.add(int(wid))
                        except (ValueError, TypeError):
                            pass
                except json.JSONDecodeError:
                    pass
        if done_ids:
            _log(f"[dispatch] {len(done_ids)} window(s) already in {out_path} — skipping")

    pending = [p for p in payloads if p["window"]["window_id"] not in done_ids]
    if not pending:
        _log("[dispatch] Nothing to do — all windows already completed.")
        if smoke:
            _smoke_sent = {
                "window_id": "smoke_sentinel",
                "smoke": True,
                "note": "smoke run: all windows already present in done_ids (should not happen after truncate)",
            }
            with open(out_path, "a") as _sf:
                _sf.write(json.dumps(_smoke_sent) + "\n")
        return

    _log(f"[dispatch] Submitting {len(pending)} window(s) via run_window ...")
    print(f"[dispatch] Submitting {len(pending)} window(s) via run_window ...", flush=True)

    # ── Fan out, stream results line-by-line ──────────────────────────────────
    n_records = 0
    n_windows_done = 0

    map_exc = None  # type: ignore[assignment]
    try:
        with open(out_path, "a") as f:
            try:
                for records in run_window.map(pending, return_exceptions=True):
                    n_windows_done += 1
                    if isinstance(records, Exception):
                        msg = (
                            f"[dispatch] WARNING: window {n_windows_done}/{len(pending)} failed — "
                            f"{type(records).__name__}: {records}"
                        )
                        _log(msg)
                        print(msg, flush=True)
                        continue
                    if len(records) == 0:
                        w_id = pending[n_windows_done - 1]["window"]["window_id"]
                        msg = (
                            f"[dispatch] WARNING: window {w_id} returned 0 records — "
                            f"no valid regimes (data gap or gene filter)"
                        )
                        _log(msg)
                        print(msg, flush=True)
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")
                        n_records += 1
                    f.flush()
                    progress = (
                        f"[dispatch] {n_windows_done}/{len(pending)} windows done | "
                        f"{n_records} regime records so far"
                    )
                    _log(progress)
                    print(progress, flush=True)
            except Exception as exc:
                err_msg = (
                    f"[dispatch] ERROR: run_window() raised: {type(exc).__name__}: {exc}"
                )
                _log(err_msg)
                print(err_msg, flush=True)
                import traceback
                tb = traceback.format_exc()
                _log(tb)
                print(tb, flush=True)
                map_exc = exc
    except Exception as _outer_exc:
        # Catch errors opening out_path or other unexpected failures before/after map
        import traceback as _tb2
        err_msg = f"[dispatch] FATAL: {type(_outer_exc).__name__}: {_outer_exc}"
        print(err_msg, flush=True)
        print(_tb2.format_exc(), flush=True)
        if smoke:
            try:
                with open(out_path, "a") as _ef:
                    _ef.write(json.dumps({
                        "window_id": "smoke_error",
                        "smoke": True,
                        "error": "fan_out_failed",
                        "exc": str(_outer_exc),
                    }) + "\n")
            except Exception:
                pass
        raise
    finally:
        try:
            _log_fh.close()
        except Exception:
            pass
        # ── Smoke summary sentinel ─────────────────────────────────────────────
        # In finally so it runs even when a BaseException (SystemExit,
        # KeyboardInterrupt, Modal signal) bypasses the except Exception blocks.
        if smoke and n_records == 0:
            try:
                smoke_summary = {
                    "window_id": "smoke_summary",
                    "smoke": True,
                    "n_windows": n_windows_done,
                    "n_records": 0,
                    "note": "smoke run completed; 0 regime records (all windows returned empty)",
                }
                with open(out_path, "a") as _sf:
                    _sf.write(json.dumps(smoke_summary) + "\n")
                msg = f"[dispatch] smoke summary sentinel written to {out_path}"
                print(msg, flush=True)
                try:
                    with open(_log_path, "a") as _lf:
                        _lf.write(msg + "\n")
                except Exception:
                    pass
            except Exception as _sf_exc:
                print(f"[dispatch] WARN: smoke_summary sentinel write failed: {_sf_exc}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    # Smoke / capped runs (smoke=True or max_windows > 0) may legitimately
    # produce 0 regime records if all candidates fail the val gate — that is a
    # signal quality problem, not a pipeline failure.  Only treat 0 records as
    # an error in a full production run.
    _zero_records_is_error = n_records == 0 and not smoke and not max_windows
    exit_code = 1 if (_zero_records_is_error or (map_exc is not None and not smoke)) else 0
    summary_lines = [
        "",
        "[dispatch] Complete.",
        f"  Windows submitted : {len(pending)}",
        f"  Regime records    : {n_records}",
        f"  Output            : {out_path}",
    ]
    if pending:
        summary_lines.append(f"  Records/window    : {n_records / len(pending):.1f} avg")
    if map_exc is not None:
        summary_lines.append(f"  map_exc           : {type(map_exc).__name__}: {map_exc}")
    summary_lines.append(f"exit_code={exit_code}")

    with open(_log_path, "a") as _lf:
        for line in summary_lines:
            if line:
                _lf.write(line + "\n")

    for line in summary_lines:
        if line:
            print(line, flush=True)

    if exit_code != 0:
        sys.exit(exit_code)


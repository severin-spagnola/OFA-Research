"""
RunPod Serverless Handler — Walk-Forward Window
=================================================
Receives a single walk-forward window (train_days + val_days as date lists),
evaluates n_candidates gene combinations, returns regime records.

This is the parallelization unit: the orchestrator dispatches one job per window,
RunPod evaluates them in parallel across workers.

Job payload:
{
    "input": {
        "job": "wf_window",
        "window_id": 0,
        "train_days": ["2024-03-01", "2024-03-04", ...],
        "val_days": ["2024-06-10", "2024-06-11", ...],
        "n_candidates": 1000,
        "top_n": 20,
        "min_trades": 30,
        "seed": 42,
        "chain_subdir": "options_5dte"
    }
}
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import time as tm
import traceback
import urllib.request
from pathlib import Path

import pandas as pd

# Set up paths — walk-forward code is in /app/current/options_wf/
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "current" / "options_wf"))

# Use all available CPUs
if not os.environ.get("OPTIONS_WORKERS"):
    os.environ["OPTIONS_WORKERS"] = str(mp.cpu_count())


def _print_build_info():
    sha = os.environ.get("BUILD_GIT_SHA", "unknown")
    bust = os.environ.get("BUILD_CACHE_BUST", "?")
    print(f"[BUILD] OFA-Research WF options | git={sha} cache_bust={bust}", flush=True)

_print_build_info()


def _read_prefix(path: Path, n: int = 100) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _is_gzip_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 2:
        return False
    return _read_prefix(path, 2) == b"\x1f\x8b"


def _download_with_urllib(url: str, out_path: Path, timeout_sec: int = 300) -> dict:
    headers = {
        "User-Agent": "runpod-wf-worker/1.0",
        "Accept": "application/gzip, application/octet-stream, */*",
        "Connection": "close",
        "Cache-Control": "no-cache",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        status = getattr(resp, "status", None)
        content_type = resp.headers.get("Content-Type", "")
        content_length = resp.headers.get("Content-Length", "")
        with open(out_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    return {
        "method": "urllib",
        "status": status,
        "content_type": content_type,
        "content_length": content_length,
    }


def _download_with_curl(url: str, out_path: Path, timeout_sec: int = 600) -> dict:
    cmd = [
        "curl", "-fL", "--max-time", str(timeout_sec),
        "-H", "Accept: application/gzip, application/octet-stream, */*",
        "-H", "Cache-Control: no-cache",
        "-A", "runpod-wf-worker/1.0",
        "-o", str(out_path),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
    return {"method": "curl", "status": "n/a"}


def _ensure_options_data() -> None:
    """Download options chain + underlying data from Render if not present."""
    data_dir = Path(os.environ.get("OPTIONS_DATA_DIR", "/data"))
    marker = data_dir / ".options_data_v1"

    if marker.exists():
        und_dir = data_dir / "underlying"
        chain_dir = data_dir / "options_5dte"
        n_und = sum(1 for _ in und_dir.glob("*.parquet")) if und_dir.exists() else 0
        n_chain = sum(1 for _ in chain_dir.glob("*.parquet")) if chain_dir.exists() else 0
        if n_und >= 100 and n_chain >= 100:
            return
        print(f"[data] Marker exists but data incomplete (und={n_und}, chain={n_chain}); refreshing", flush=True)
        marker.unlink(missing_ok=True)

    render_url = os.environ.get("RENDER_DATA_URL",
                                 "https://gap-autotrader-cxt8.onrender.com")
    render_pw = os.environ.get("RENDER_DATA_PASSWORD", "GOON")
    url = f"{render_url}/download-data?password={render_pw}&path=options/options_data.tar.gz"
    tar_path = Path("/tmp/options_data.tar.gz")

    print("[data] Downloading options data from Render...", flush=True)
    t0 = tm.time()
    errors = []

    for attempt in range(2):
        method = "urllib" if attempt == 0 else "curl"
        try:
            tar_path.unlink(missing_ok=True)
            if method == "urllib":
                meta = _download_with_urllib(url, tar_path, timeout_sec=300)
            else:
                meta = _download_with_curl(url, tar_path, timeout_sec=600)

            size_bytes = tar_path.stat().st_size if tar_path.exists() else 0
            print(f"[data] Attempt {attempt+1}/2 via {method} bytes={size_bytes}", flush=True)

            if not _is_gzip_file(tar_path):
                prefix = _read_prefix(tar_path, 100) if tar_path.exists() else b""
                raise RuntimeError(f"Not gzip: first_100_bytes={prefix!r}")

            data_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            tar_path.unlink(missing_ok=True)

            und_dir = data_dir / "underlying"
            chain_dir = data_dir / "options_5dte"
            n_und = sum(1 for _ in und_dir.glob("*.parquet")) if und_dir.exists() else 0
            n_chain = sum(1 for _ in chain_dir.glob("*.parquet")) if chain_dir.exists() else 0

            if n_und < 100 or n_chain < 100:
                raise RuntimeError(f"Insufficient data: und={n_und}, chain={n_chain}")

            marker.touch()
            elapsed = tm.time() - t0
            print(f"[data] Ready: {n_und} underlying, {n_chain} chains ({elapsed:.1f}s)", flush=True)
            return

        except Exception as e:
            errors.append(f"{method}: {e}")
            print(f"[data] Attempt {attempt+1}/2 FAILED: {e}", flush=True)
            tar_path.unlink(missing_ok=True)
            if attempt == 0:
                tm.sleep(1.0)

    raise RuntimeError("FAILED to download options data: " + " | ".join(errors))


def handler(event: dict) -> dict:
    """RunPod serverless handler — walk-forward window evaluation."""
    try:
        timings = {"handler_start": tm.time()}

        _ensure_options_data()
        timings["data_ready"] = tm.time()

        inp = event.get("input", event)
        job_type = inp.get("job", "wf_window")

        if job_type == "wf_window":
            result = _handle_wf_window(inp)
        elif job_type == "diagnostic":
            result = _handle_diagnostic(inp)
        else:
            return {"error": f"Unknown job type: {job_type}"}

        timings["job_done"] = tm.time()
        result["timings"] = {
            "data_download_sec": round(timings["data_ready"] - timings["handler_start"], 1),
            "total_sec": round(timings["job_done"] - timings["handler_start"], 1),
        }
        result["worker"] = {
            "cpu_count": mp.cpu_count(),
            "workers_used": int(os.environ.get("OPTIONS_WORKERS", mp.cpu_count())),
        }
        return result

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def _handle_wf_window(inp: dict) -> dict:
    """Evaluate a single walk-forward window."""
    from datetime import date as _date
    from wf_runner import WFWindow, evaluate_window, N_CONTEXT_DAYS
    from options_backtest import (
        CostModel, load_underlying, load_chain, load_chain_meta,
        get_available_days,
    )

    window_id = inp.get("window_id", 0)
    train_day_strs = inp.get("train_days", [])
    val_day_strs = inp.get("val_days", [])
    n_candidates = inp.get("n_candidates", 1000)
    top_n = inp.get("top_n", 20)
    min_trades = inp.get("min_trades", 30)
    seed = inp.get("seed", 42)
    chain_subdir = inp.get("chain_subdir", "options_5dte")

    data_dir = Path(os.environ.get("OPTIONS_DATA_DIR", "/data"))

    # Parse dates
    train_days = sorted(_date.fromisoformat(s) for s in train_day_strs)
    val_days = sorted(_date.fromisoformat(s) for s in val_day_strs)

    if not train_days or not val_days:
        return {"error": "Empty train_days or val_days"}

    print(f"[wf] Window {window_id}: train {train_days[0]}→{train_days[-1]} "
          f"({len(train_days)}d), val {val_days[0]}→{val_days[-1]} ({len(val_days)}d)",
          flush=True)
    print(f"[wf] Candidates: {n_candidates}, top_n: {top_n}, seed: {seed}", flush=True)

    # Load all available days first — needed for both fwd_days and context_days.
    available = get_available_days(data_dir, chain_subdir)
    first_train = train_days[0]

    # fwd_days: all available days after val_end — the forward phase walks
    # through these until kill conditions fire.
    fwd_days = sorted(d for d in available if d > val_days[-1])

    window = WFWindow(
        window_id=window_id,
        train_start=train_days[0],
        train_end=train_days[-1],
        val_start=val_days[0],
        val_end=val_days[-1],
        train_days=train_days,
        val_days=val_days,
        fwd_days=fwd_days,
    )

    # Discover context days: N_CONTEXT_DAYS before the first train day.
    # These provide prev_close/prev_range data for the first train day's
    # gap and regime filter signals, ensuring parity with local runs.
    prior_days = [d for d in available if d < first_train]
    context_days = prior_days[-N_CONTEXT_DAYS:]
    print(f"[wf] Context days for prev_close: {len(context_days)} "
          f"({context_days[0] if context_days else 'none'}→"
          f"{context_days[-1] if context_days else 'none'})", flush=True)

    # Pre-load data for context + window days
    t0 = tm.time()
    all_load_days = sorted(context_days) + train_days + val_days
    preloaded_und = {}
    preloaded_chains = {}
    preloaded_metas = {}

    for d in all_load_days:
        und = load_underlying(data_dir, d)
        if not und.empty:
            preloaded_und[d] = und
        chain = load_chain(data_dir, d, chain_subdir)
        if not chain.empty:
            preloaded_chains[d] = chain
        meta = load_chain_meta(data_dir, d, chain_subdir)
        if meta:
            preloaded_metas[d] = meta

    preload_sec = tm.time() - t0
    print(f"[wf] Pre-loaded {len(preloaded_und)} und, {len(preloaded_chains)} chains "
          f"in {preload_sec:.1f}s", flush=True)

    # Evaluate window
    t0 = tm.time()
    records = evaluate_window(
        window=window,
        data_dir=data_dir,
        chain_subdir=chain_subdir,
        n_candidates=n_candidates,
        top_n=top_n,
        min_trades=min_trades,
        seed=seed,
        cost=CostModel(),
        preloaded_und=preloaded_und,
        preloaded_chains=preloaded_chains,
        preloaded_metas=preloaded_metas,
        context_days=context_days,
    )
    eval_sec = tm.time() - t0

    n_profitable = sum(1 for r in records if r.get("forward_profitable", False))
    print(f"[wf] Window {window_id}: {len(records)} val-tested, "
          f"{n_profitable} profitable ({eval_sec:.1f}s)", flush=True)

    return {
        "status": "ok",
        "window_id": window_id,
        "n_regimes": len(records),
        "n_profitable": n_profitable,
        "regimes": records,
        "preload_sec": round(preload_sec, 1),
        "eval_sec": round(eval_sec, 1),
    }


def _handle_diagnostic(inp: dict) -> dict:
    """Quick diagnostic — verify data, imports, paths."""
    from options_backtest import get_available_days
    from options_genes import ENTRY_CONDITIONS, TRADE_TYPES, REGIME_FILTERS

    data_dir = Path(os.environ.get("OPTIONS_DATA_DIR", "/data"))
    chain_subdir = inp.get("chain_subdir", "options_5dte")

    days = get_available_days(data_dir, chain_subdir)

    return {
        "status": "ok",
        "job": "diagnostic",
        "build": {
            "git_sha": os.environ.get("BUILD_GIT_SHA", "unknown"),
            "cache_bust": os.environ.get("BUILD_CACHE_BUST", "?"),
        },
        "available_days": len(days),
        "date_range": [str(days[0]), str(days[-1])] if days else [],
        "genes": {
            "n_entry_conditions": len(ENTRY_CONDITIONS),
            "n_trade_types": len(TRADE_TYPES),
            "n_regime_filters": len(REGIME_FILTERS),
        },
        "environment": {
            "python_version": sys.version,
            "cpu_count": mp.cpu_count(),
        },
    }


# ─── RunPod entry point ──────────────────────────────────────────────────────

import runpod

runpod.serverless.start({"handler": handler})

"""
RunPod Serverless Handler — OFA-Research (Overnight Mode)
=========================================================
Stripped-down handler for overnight MES futures strategy search.
No GEX/options/classifier — just MES 1m data + gene combinator.

Deploy:
  1. Build Docker image with Dockerfile
  2. Push to registry, create RunPod serverless endpoint
  3. Submit jobs via orchestrator.py or RunPod API
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import time as tm
import traceback
import urllib.request
from pathlib import Path

# Set up paths
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "current" / "overfit_finder"))
sys.path.insert(0, str(_REPO_ROOT / "archived" / "fvg_gap"))

# Set parallel workers from env
if not os.environ.get("OVERFIT_WORKERS"):
    import multiprocessing as mp
    os.environ["OVERFIT_WORKERS"] = str(mp.cpu_count())


def _print_build_info():
    sha = os.environ.get("BUILD_GIT_SHA", "unknown")
    bust = os.environ.get("BUILD_CACHE_BUST", "?")
    print(f"[BUILD] OFA-Research overnight | git={sha} cache_bust={bust}", flush=True)

_print_build_info()


def _ensure_data(asset: str = "MES") -> None:
    """Download 1m bar data from Render if not already present."""
    ticker_lower = asset.lower()

    if asset == "MES":
        data_path = Path(os.environ.get("MES_DATA_PATH", "/data/mes/mes_1m.csv"))
    else:
        env_key = f"{asset.upper()}_DATA_PATH"
        data_path = Path(os.environ.get(env_key, f"/data/{ticker_lower}/{ticker_lower}_1m.csv"))

    if data_path.exists() and data_path.stat().st_size > 0:
        return

    render_url = os.environ.get("RENDER_DATA_URL", "")
    render_pw = os.environ.get("RENDER_DATA_PASSWORD", "")
    if not render_url:
        raise RuntimeError(
            f"{asset} data not found and RENDER_DATA_URL not set. "
            "Set RENDER_DATA_URL=https://gap-autotrader-cxt8.onrender.com"
        )

    gz_filename = f"{ticker_lower}_1m.csv.gz"
    url = f"{render_url}/download-data?password={render_pw}&path={ticker_lower}/{gz_filename}"
    gz_path = data_path.parent / gz_filename
    data_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data] Downloading {asset} data from Render...")
    t0 = tm.time()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(resp, f)

    # Decompress
    with gzip.open(gz_path, "rb") as gz_in, open(data_path, "wb") as csv_out:
        shutil.copyfileobj(gz_in, csv_out)
    gz_path.unlink()

    size_mb = data_path.stat().st_size / 1024 / 1024
    print(f"[data] Downloaded and decompressed {size_mb:.0f}MB in {tm.time()-t0:.1f}s")


def _build_progress_callback(job_id: str):
    """Build a callback that POSTs regime progress to Render."""
    render_url = os.environ.get("RENDER_DATA_URL", "")
    if not render_url or not job_id:
        return None

    def callback(regime_index: int, regime_elapsed_sec: float,
                 regime_record: dict | None = None):
        try:
            url = f"{render_url}/regime-progress"
            payload = json.dumps({
                "job_id": job_id,
                "regime_index": regime_index,
                "regime_elapsed_sec": round(regime_elapsed_sec, 1),
                "regime_record": regime_record,
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[progress] Failed to POST regime progress: {e}")

    return callback


def handler(event: dict) -> dict:
    """RunPod serverless handler — overnight mode only.

    Input payload:
    {
        "job": "llm_walkforward",
        "asset": "MES",
        "wf_window": 45,
        "wf_max_regime": 180,
        "wf_start": "2024-06-01",
        "n_strategies": 10,
        "n_refinements": 10,
        "max_dd_dollars": 3500,
        "use_seeds": false,
        "use_genes": true,
        "max_regimes": 6,
        "meta_config": { ... }    // optional
    }
    """
    try:
        import multiprocessing as _mp
        timings = {"handler_start": tm.time()}

        inp = event.get("input", event)
        asset = inp.get("asset", "MES")
        job = inp.get("job", "llm_walkforward")
        runpod_job_id = event.get("id", "")

        _ensure_data(asset)
        timings["data_ready"] = tm.time()
        t0 = tm.time()

        if job == "llm_walkforward":
            result = _handle_llm_walkforward(inp, t0, asset=asset,
                                             job_id=runpod_job_id)
        elif job == "diagnostic":
            result = _handle_diagnostic(inp, t0)
        else:
            return {"error": f"Unknown job type: {job}. OFA-Research only supports llm_walkforward and diagnostic."}

        timings["job_done"] = tm.time()

        result["timings"] = {
            "data_download_sec": round(timings["data_ready"] - timings["handler_start"], 1),
            "job_sec": round(timings["job_done"] - timings["data_ready"], 1),
            "total_sec": round(timings["job_done"] - timings["handler_start"], 1),
        }
        result["worker"] = {
            "cpu_count": _mp.cpu_count(),
            "workers_used": int(os.environ.get("OVERFIT_WORKERS", _mp.cpu_count())),
        }
        result["mode"] = "overnight"
        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _handle_llm_walkforward(inp: dict, t0: float, asset: str = "MES",
                            job_id: str = "") -> dict:
    """Run walk-forward with gene combinator — overnight mode forced ON."""
    from overfit_search import (
        run_walkforward_llm, print_walkforward_summary,
        save_walkforward_results, OUTPUT_DIR, MetaConfig,
    )

    # Build MetaConfig from job payload if provided
    meta = None
    if inp.get("meta_config"):
        meta = MetaConfig.from_dict(inp["meta_config"])

    # Build progress callback
    progress_cb = _build_progress_callback(job_id)

    # OVERNIGHT = TRUE always — this is the overnight research worker
    wf_summary = run_walkforward_llm(
        window_days=inp.get("wf_window", 45),
        max_regime_days=inp.get("wf_max_regime", 180),
        overlap_days=inp.get("wf_overlap", 0),
        n_strategies=inp.get("n_strategies", 10),
        n_refinements=inp.get("n_refinements", 10),
        start_date=inp.get("wf_start"),
        max_dd_dollars=inp.get("max_dd_dollars", 3500),
        use_seeds=inp.get("use_seeds", False),
        use_genes=inp.get("use_genes", True),
        max_regimes=inp.get("max_regimes", 6),
        meta=meta,
        asset=asset,
        on_regime_done=progress_cb,
        overnight=True,  # <-- ALWAYS overnight
    )

    print_walkforward_summary(wf_summary)
    results_path = save_walkforward_results(wf_summary, OUTPUT_DIR)

    output = wf_summary.to_dict()
    output["status"] = "ok"
    output["job"] = "llm_walkforward"
    output["asset"] = asset
    output["mode"] = "overnight"
    output["elapsed_sec"] = round(tm.time() - t0, 1)
    output["results_path"] = str(results_path)

    # Attach uncensored MFE/MAE records if collected
    from overfit_search import _UNCENSORED_RECORDS
    if _UNCENSORED_RECORDS:
        output["uncensored_mfe_mae"] = _UNCENSORED_RECORDS
        print(f"  [uncensored] Returning {len(_UNCENSORED_RECORDS)} records")

    return output


def _handle_diagnostic(inp: dict, t0: float) -> dict:
    """Quick diagnostic — verify data, imports, overnight gene counts."""
    diag = {"status": "ok", "job": "diagnostic", "mode": "overnight"}

    # Data
    mes_path = Path(os.environ.get("MES_DATA_PATH", "/data/mes/mes_1m.csv"))
    diag["data"] = {
        "mes_exists": mes_path.exists(),
        "mes_size_mb": round(mes_path.stat().st_size / 1024 / 1024, 1) if mes_path.exists() else 0,
    }

    # Gene combinator overnight counts
    try:
        from gene_combinator import (
            OVERNIGHT_ENTRY_GENES, OVERNIGHT_FILTER_ENTRY_GENES,
            OVERNIGHT_TIME_WINDOWS, ENTRY_GENES, FILTER_ENTRY_GENES,
        )
        diag["genes"] = {
            "rth_entry_genes": len(ENTRY_GENES),
            "overnight_entry_genes": len(OVERNIGHT_ENTRY_GENES),
            "rth_filter_genes": len(FILTER_ENTRY_GENES),
            "overnight_filter_genes": len(OVERNIGHT_FILTER_ENTRY_GENES),
            "overnight_time_windows": len(OVERNIGHT_TIME_WINDOWS),
        }
    except Exception as e:
        diag["genes"] = {"error": str(e)}

    # Strategy DSL overnight support
    try:
        from strategy_dsl import _build_overnight_sessions, _run_strategy_overnight
        diag["strategy_dsl"] = {"overnight_support": True}
    except ImportError as e:
        diag["strategy_dsl"] = {"overnight_support": False, "error": str(e)}

    diag["environment"] = {
        "python_version": sys.version,
        "OVERFIT_WORKERS": os.environ.get("OVERFIT_WORKERS", ""),
        "RENDER_DATA_URL": os.environ.get("RENDER_DATA_URL", "")[:30] + "...",
    }

    diag["elapsed_sec"] = round(tm.time() - t0, 1)
    return diag


# ─── RunPod entry point ─────────────────────────────────────────────────────

import runpod

runpod.serverless.start({"handler": handler})

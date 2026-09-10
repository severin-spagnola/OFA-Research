"""
RunPod Serverless Handler — Penny Gapper Gene Search
=====================================================
Receives a job with profile + seed, evaluates gene candidates against
baked-in bar cache data, returns top results.

Deploy:
  1. Build Docker image with Dockerfile.penny
  2. Push to registry, create RunPod serverless CPU endpoint
  3. Submit jobs via penny_orchestrator.py

Job payload:
{
    "input": {
        "profile": "listed_highvol",    # profile name
        "candidates": 5000,             # genes to evaluate per worker
        "seed": 42,                     # RNG seed (different per worker)
        "val_split": 0.30,              # fraction for validation
        "top_n": 50,                    # how many to validate
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

# Set up paths — penny code is in /app/current/penny_research/
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "current" / "penny_research"))

# Use all available CPUs
if not os.environ.get("PENNY_WORKERS"):
    os.environ["PENNY_WORKERS"] = str(mp.cpu_count())


def _print_build_info():
    sha = os.environ.get("BUILD_GIT_SHA", "unknown")
    bust = os.environ.get("BUILD_CACHE_BUST", "?")
    print(f"[BUILD] OFA-Research penny search | git={sha} cache_bust={bust}", flush=True)

_print_build_info()


def _read_prefix(path: Path, n: int = 100) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _is_gzip_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 2:
        return False
    return _read_prefix(path, 2) == b"\x1f\x8b"


def _download_with_urllib(url: str, out_path: Path, timeout_sec: int = 120) -> dict:
    headers = {
        "User-Agent": "runpod-penny-worker/1.0",
        "Accept": "application/gzip, application/octet-stream, */*",
        "Connection": "close",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
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


def _download_with_curl(url: str, out_path: Path, timeout_sec: int = 180) -> dict:
    cmd = [
        "curl", "-fL", "--max-time", str(timeout_sec),
        "-H", "Accept: application/gzip, application/octet-stream, */*",
        "-H", "Cache-Control: no-cache",
        "-A", "runpod-penny-worker/1.0",
        "-o", str(out_path),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
    return {"method": "curl", "status": "n/a", "content_type": "n/a", "content_length": "n/a"}


def _clear_penny_cache(data_dir: Path, marker: Path) -> None:
    marker.unlink(missing_ok=True)
    bar_cache_dir = data_dir / "bar_cache"
    if bar_cache_dir.exists():
        shutil.rmtree(bar_cache_dir, ignore_errors=True)


def _ensure_penny_data() -> None:
    """Download penny bar cache from Render if not already present."""
    data_dir = _REPO_ROOT / "current" / "penny_research" / "data"
    marker = data_dir / ".penny_data_v5"
    bar_cache_dir = data_dir / "bar_cache"
    min_bar_files = int(os.environ.get("PENNY_MIN_BAR_FILES", "1000"))

    if marker.exists():
        n_existing = sum(1 for _ in bar_cache_dir.glob("*.parquet")) if bar_cache_dir.exists() else 0
        if n_existing >= min_bar_files:
            return
        print(
            f"[data] Marker exists but cache looks incomplete ({n_existing} files < {min_bar_files}); forcing refresh",
            flush=True,
        )
        _clear_penny_cache(data_dir, marker)

    render_url = os.environ.get("RENDER_DATA_URL",
                                 "https://gap-autotrader-cxt8.onrender.com")
    render_pw = os.environ.get("RENDER_DATA_PASSWORD", "GOON")

    url = f"{render_url}/download-data?password={render_pw}&path=penny/penny_bar_cache.tar.gz"
    tar_path = Path("/tmp/penny_bar_cache.tar.gz")

    print("[data] Downloading penny bar cache from Render...", flush=True)
    t0 = tm.time()
    errors = []
    for attempt in range(2):
        method = "urllib" if attempt == 0 else "curl"
        try:
            tar_path.unlink(missing_ok=True)

            if method == "urllib":
                meta = _download_with_urllib(url, tar_path, timeout_sec=120)
            else:
                meta = _download_with_curl(url, tar_path, timeout_sec=180)

            size_bytes = tar_path.stat().st_size if tar_path.exists() else 0
            print(
                f"[data] Attempt {attempt+1}/2 via {meta['method']} "
                f"status={meta['status']} content_type={meta['content_type']} "
                f"content_length={meta['content_length']} bytes={size_bytes}",
                flush=True,
            )

            if not _is_gzip_file(tar_path):
                prefix = _read_prefix(tar_path, 100) if tar_path.exists() else b""
                raise RuntimeError(
                    "Downloaded file is not gzip. "
                    f"first_100_bytes={prefix!r}"
                )

            data_dir.mkdir(parents=True, exist_ok=True)
            if bar_cache_dir.exists():
                shutil.rmtree(bar_cache_dir, ignore_errors=True)

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            tar_path.unlink(missing_ok=True)

            n_files = sum(1 for _ in bar_cache_dir.glob("*.parquet")) if bar_cache_dir.exists() else 0
            if n_files < min_bar_files:
                raise RuntimeError(
                    f"Extracted bar cache has too few files: {n_files} < {min_bar_files}"
                )

            # Download survey parquet (ticker-day metadata)
            survey_path = data_dir / "phase3_full.parquet"
            if not survey_path.exists():
                survey_url = f"{render_url}/download-data?password={render_pw}&path=penny/phase3_full.parquet"
                survey_tmp = Path("/tmp/phase3_full.parquet")
                print("[data] Downloading survey parquet from Render...", flush=True)
                try:
                    _download_with_urllib(survey_url, survey_tmp, timeout_sec=30)
                except Exception:
                    _download_with_curl(survey_url, survey_tmp, timeout_sec=30)
                shutil.move(str(survey_tmp), str(survey_path))
                print(f"[data] Survey parquet: {survey_path.stat().st_size / 1024:.0f}KB", flush=True)

            marker.touch()
            size_mb = sum(f.stat().st_size for f in bar_cache_dir.iterdir()) / 1024 / 1024
            print(
                f"[data] Extracted {n_files} bar cache files ({size_mb:.0f}MB) in {tm.time()-t0:.1f}s",
                flush=True,
            )
            return
        except Exception as e:
            errors.append(f"{method}: {e}")
            print(f"[data] Attempt {attempt+1}/2 FAILED: {e}", flush=True)
            _clear_penny_cache(data_dir, marker)
            tar_path.unlink(missing_ok=True)
            if attempt == 0:
                tm.sleep(1.0)
                continue

    raise RuntimeError("FAILED to download penny data after retries: " + " | ".join(errors))


def handler(event: dict) -> dict:
    """RunPod serverless handler — penny gene search."""
    try:
        timings = {"handler_start": tm.time()}

        _ensure_penny_data()
        timings["data_ready"] = tm.time()

        inp = event.get("input", event)
        job_type = inp.get("job", "penny_search")
        runpod_job_id = event.get("id", "")

        if job_type == "penny_search":
            result = _handle_penny_search(inp)
        elif job_type == "diagnostic":
            result = _handle_diagnostic(inp)
        else:
            return {"error": f"Unknown job type: {job_type}"}

        timings["job_done"] = tm.time()
        result["timings"] = {
            "data_download_sec": round(timings.get("data_ready", timings["handler_start"]) - timings["handler_start"], 1),
            "total_sec": round(timings["job_done"] - timings["handler_start"], 1),
        }
        result["worker"] = {
            "cpu_count": mp.cpu_count(),
            "workers_used": int(os.environ.get("PENNY_WORKERS", mp.cpu_count())),
        }
        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _handle_penny_search(inp: dict) -> dict:
    """Run penny gene search for one profile + seed."""
    from penny_genes import sample_candidates, describe_genes, PennyGenes
    from penny_backtest import evaluate_pooled, preload_bar_data, BacktestResult
    from penny_fitness import compute_fitness, FitnessResult
    from penny_search import load_ticker_days, parallel_eval, BAR_CACHE_DIR

    profile = inp.get("profile", "listed_highvol")
    n_candidates = inp.get("candidates", 5000)
    seed = inp.get("seed", 42)
    val_split = inp.get("val_split", 0.30)
    top_n = inp.get("top_n", 50)
    n_workers = int(os.environ.get("PENNY_WORKERS", mp.cpu_count()))

    print(f"[penny] Profile={profile} candidates={n_candidates} seed={seed} workers={n_workers}", flush=True)

    # Load ticker-day metadata
    t0 = tm.time()
    ticker_days = load_ticker_days(profile)
    print(f"[penny] Loaded {len(ticker_days)} ticker-days ({tm.time()-t0:.1f}s)", flush=True)

    if len(ticker_days) < 30:
        return {"error": f"Only {len(ticker_days)} ticker-days for {profile}, need >= 30"}

    # Debug: check what files exist vs what we're looking for
    bar_cache_path = Path(str(BAR_CACHE_DIR))
    actual_files = sorted(bar_cache_path.glob("*.parquet"))[:5] if bar_cache_path.exists() else []
    expected_keys = [f"{td['ticker']}_{td['date']}.parquet" for td in ticker_days[:5]]
    print(f"[penny] BAR_CACHE_DIR={BAR_CACHE_DIR} exists={bar_cache_path.exists()} "
          f"files={len(list(bar_cache_path.glob('*.parquet'))) if bar_cache_path.exists() else 0}", flush=True)
    print(f"[penny] Sample actual files: {[f.name for f in actual_files]}", flush=True)
    print(f"[penny] Sample expected keys: {expected_keys}", flush=True)
    # Deep debug: check first ticker-day file existence directly
    if ticker_days:
        td0 = ticker_days[0]
        test_key = f"{td0['ticker']}_{td0['date']}"
        test_path = bar_cache_path / f"{test_key}.parquet"
        print(f"[penny] DEBUG: td0 ticker={repr(td0['ticker'])} date={repr(td0['date'])} "
              f"types=({type(td0['ticker']).__name__}, {type(td0['date']).__name__})", flush=True)
        print(f"[penny] DEBUG: test_path={test_path} exists={test_path.exists()}", flush=True)
        # Try listing files that start with this ticker
        import os as _os
        matches = [f for f in _os.listdir(str(bar_cache_path)) if f.startswith(td0['ticker'] + '_')]
        print(f"[penny] DEBUG: os.listdir matches for {td0['ticker']}_*: {matches[:3]}", flush=True)

    # Pre-load ALL bar data into memory once (avoids re-reading parquet per candidate)
    # Use direct glob instead of ticker_day matching to bypass any path/name issues
    t0 = tm.time()
    bar_data = {}
    n_load_err = 0
    first_load_err = None
    for pf in bar_cache_path.glob("*.parquet"):
        key = pf.stem  # e.g. "RGC_2025-03-18"
        try:
            df = pd.read_parquet(pf)
            if not df.empty:
                bar_data[key] = df
        except Exception as e:
            n_load_err += 1
            if first_load_err is None:
                first_load_err = f"{pf.name}: {e}"
    preload_sec = tm.time() - t0
    mem_mb = sum(df.memory_usage(deep=True).sum() for df in bar_data.values()) / 1024 / 1024
    print(f"[penny] Pre-loaded {len(bar_data)} bar files into memory ({mem_mb:.0f}MB, {preload_sec:.1f}s) "
          f"errors={n_load_err} first_err={first_load_err}", flush=True)

    # Temporal split (by unique date boundary)
    unique_dates = sorted(set(td["date"] for td in ticker_days))
    split_date_idx = int(len(unique_dates) * (1 - val_split))
    train_cutoff = unique_dates[split_date_idx]

    train_days = [td for td in ticker_days if td["date"] < train_cutoff]
    val_days = [td for td in ticker_days if td["date"] >= train_cutoff]

    if not train_days or not val_days:
        return {"error": "Split produced empty train or val set"}

    print(f"[penny] Train: {len(train_days)} ({train_days[0]['date']} to {train_days[-1]['date']})", flush=True)
    print(f"[penny] Val:   {len(val_days)} ({val_days[0]['date']} to {val_days[-1]['date']})", flush=True)

    # Sample candidates
    candidates = sample_candidates(n=n_candidates, seed=seed)
    print(f"[penny] Sampled {len(candidates)} unique candidates", flush=True)

    # Phase 1: Evaluate all on train (parallel, in-memory data)
    print(f"[penny] Phase 1: evaluating {len(candidates)} candidates on {len(train_days)} train days ({n_workers} workers)...", flush=True)
    t0 = tm.time()
    try:
        train_results = parallel_eval(
            candidates, train_days, BAR_CACHE_DIR,
            n_workers=n_workers, bar_data=bar_data,
        )
    except Exception as e:
        print(f"[penny] ERROR in parallel train eval: {e}", flush=True)
        traceback.print_exc()
        return {"error": f"Train eval failed: {e}", "traceback": traceback.format_exc()}

    train_elapsed = tm.time() - t0
    viable = [(idx, fr, sm) for idx, fr, sm in train_results if fr.fitness > 0]

    # Diagnostic: trade counts across all candidates
    trade_counts = [sm.get("n_trades", 0) for _, _, sm in train_results]
    n_zero = sum(1 for t in trade_counts if t == 0)
    n_any = sum(1 for t in trade_counts if t > 0)
    max_trades = max(trade_counts) if trade_counts else 0
    avg_trades = sum(trade_counts) / len(trade_counts) if trade_counts else 0
    print(f"[penny] Trade diagnostic: {n_any} with trades, {n_zero} with zero trades, "
          f"avg={avg_trades:.1f}, max={max_trades}", flush=True)

    # Show fitness distribution for candidates that had trades
    if n_any > 0:
        fitnesses = [fr.fitness for _, fr, sm in train_results if sm.get("n_trades", 0) > 0]
        n_positive = sum(1 for f in fitnesses if f > 0)
        n_negative = sum(1 for f in fitnesses if f <= 0)
        best_fit = max(fitnesses) if fitnesses else 0
        worst_fit = min(fitnesses) if fitnesses else 0
        print(f"[penny] Fitness (traded only): {n_positive} positive, {n_negative} non-positive, "
              f"best={best_fit:.4f}, worst={worst_fit:.4f}", flush=True)

    print(f"[penny] Train eval: {len(viable)} viable / {len(candidates)} total "
          f"({train_elapsed:.0f}s, {len(candidates)/max(train_elapsed,0.1):.0f} cand/s)", flush=True)

    # Build diagnostic dict for response
    diag = {
        "n_with_trades": n_any,
        "n_zero_trades": n_zero,
        "avg_trades": round(avg_trades, 1),
        "max_trades": max_trades,
        "preload_files": len(bar_data),
        "preload_errors": n_load_err,
        "preload_first_error": str(first_load_err)[:200] if first_load_err else None,
        "preload_sec": round(preload_sec, 1),
        "train_eval_sec": round(train_elapsed, 1),
    }
    if n_any > 0:
        fitnesses = [fr.fitness for _, fr, sm in train_results if sm.get("n_trades", 0) > 0]
        diag["n_positive_fitness"] = sum(1 for f in fitnesses if f > 0)
        diag["n_negative_fitness"] = sum(1 for f in fitnesses if f <= 0)
        diag["best_fitness"] = round(max(fitnesses), 4)
        diag["worst_fitness"] = round(min(fitnesses), 4)

    if not viable:
        return {
            "status": "ok",
            "profile": profile,
            "seed": seed,
            "n_candidates": len(candidates),
            "n_viable": 0,
            "top_results": [],
            "diagnostic": diag,
        }

    # Sort by train fitness, take top N for val
    viable.sort(key=lambda x: -x[1].fitness)
    top_viable = viable[:top_n]
    top_candidates = [candidates[idx] for idx, _, _ in top_viable]

    # Phase 2: Evaluate top on val (parallel, in-memory data)
    print(f"[penny] Phase 2: evaluating {len(top_candidates)} top candidates on {len(val_days)} val days...", flush=True)
    t0 = tm.time()
    try:
        val_results = parallel_eval(
            top_candidates, val_days, BAR_CACHE_DIR,
            n_workers=n_workers, bar_data=bar_data,
        )
    except Exception as e:
        print(f"[penny] ERROR in parallel val eval: {e}", flush=True)
        traceback.print_exc()
        return {"error": f"Val eval failed: {e}", "traceback": traceback.format_exc()}

    val_elapsed = tm.time() - t0
    print(f"[penny] Val eval: {len(val_results)} candidates ({val_elapsed:.0f}s)", flush=True)

    # Combine train + val results
    results = []
    for j, (val_idx, val_fr, val_sm) in enumerate(val_results):
        train_idx, train_fr, train_sm = top_viable[j]
        genes = candidates[train_idx]
        results.append({
            "genes": genes.to_dict(),
            "genes_desc": describe_genes(genes),
            "train_fitness": _fr_to_dict(train_fr),
            "val_fitness": _fr_to_dict(val_fr),
            "train_summary": train_sm,
            "val_summary": val_sm,
        })

    # Gate 1: val fitness must be positive
    results = [r for r in results if r["val_fitness"]["fitness"] > 0]

    # Gate 2: require positive train cum_r (no "loses on train, wins on val" flukes)
    n_before_train_gate = len(results)
    results = [r for r in results if r["train_fitness"]["cum_r"] > 0]
    n_after_train_gate = len(results)
    print(f"[penny] Gate: positive train cum_r: {n_before_train_gate} -> {n_after_train_gate}", flush=True)

    # Gate 3: 2x cost stress test on val survivors
    # Re-run val with doubled spread+slippage; reject if val cum_r goes negative
    from penny_backtest import CostModel
    stressed_cost = CostModel(
        spread_pct=0.30,       # 2x base 0.15
        slippage_pct=0.20,     # 2x base 0.10
        stop_extra_slip_pct=0.40,  # 2x base 0.20
    )
    n_before_stress = len(results)
    stress_survivors = []
    if results:
        stress_candidates = []
        stress_genes_map = []
        for r in results:
            g = PennyGenes(**{k: v for k, v in r["genes"].items()
                            if k in PennyGenes.__dataclass_fields__})
            stress_candidates.append(g)
            stress_genes_map.append(r)

        print(f"[penny] Phase 3: 2x cost stress test on {len(stress_candidates)} val survivors...", flush=True)
        t0 = tm.time()
        stress_results = parallel_eval(
            stress_candidates, val_days, BAR_CACHE_DIR,
            n_workers=n_workers, bar_data=bar_data, cost=stressed_cost,
        )
        stress_elapsed = tm.time() - t0
        print(f"[penny] Stress test: {stress_elapsed:.0f}s", flush=True)

        for j, (s_idx, s_fr, s_sm) in enumerate(stress_results):
            orig = stress_genes_map[j]
            if s_fr.cum_r > 0:
                orig["stress_val_fitness"] = _fr_to_dict(s_fr)
                stress_survivors.append(orig)
            else:
                print(f"[penny] Stress-killed: {orig['genes_desc']} "
                      f"(base val cum_r={orig['val_fitness']['cum_r']:+.1f} -> "
                      f"stress cum_r={s_fr.cum_r:+.1f})", flush=True)

        results = stress_survivors

    n_after_stress = len(results)
    print(f"[penny] Gate: 2x cost stress: {n_before_stress} -> {n_after_stress}", flush=True)

    # Rank by val fitness
    results.sort(key=lambda x: -x["val_fitness"]["fitness"])

    # Print top 5
    for i, r in enumerate(results[:5]):
        tf = r["train_fitness"]
        vf = r["val_fitness"]
        stress = r.get("stress_val_fitness", {})
        stress_info = f" stress_cum={stress.get('cum_r', '?')}" if stress else ""
        print(f"[penny] #{i+1}: val={vf['fitness']:.4f} train={tf['fitness']:.4f} "
              f"trades={tf['n_trades']}/{vf['n_trades']} "
              f"WR={tf['win_rate']*100:.0f}%/{vf['win_rate']*100:.0f}%{stress_info} "
              f"{r['genes_desc']}", flush=True)

    return {
        "status": "ok",
        "profile": profile,
        "seed": seed,
        "n_candidates": len(candidates),
        "n_viable": len(viable),
        "n_val_pass": len(results),
        "n_killed_train_gate": n_before_train_gate - n_after_train_gate,
        "n_killed_stress": n_before_stress - n_after_stress,
        "top_results": results[:20],  # return top 20
        "train_ticker_days": len(train_days),
        "val_ticker_days": len(val_days),
        "train_date_range": [train_days[0]["date"], train_days[-1]["date"]],
        "val_date_range": [val_days[0]["date"], val_days[-1]["date"]],
        "train_eval_sec": round(train_elapsed, 1),
        "val_eval_sec": round(val_elapsed, 1),
        "preload_sec": round(preload_sec, 1),
        "diagnostic": diag,
    }


def _fr_to_dict(fr) -> dict:
    """Convert FitnessResult to serializable dict."""
    return {
        "fitness": fr.fitness,
        "sharpe": fr.sharpe,
        "profit_factor": fr.profit_factor,
        "win_rate": fr.win_rate,
        "n_trades": fr.n_trades,
        "avg_r": fr.avg_r,
        "cum_r": fr.cum_r,
        "max_dd_r": fr.max_dd_r,
    }


def _handle_diagnostic(inp: dict) -> dict:
    """Quick diagnostic — verify data, imports, paths."""
    diag = {"status": "ok", "job": "diagnostic"}

    # Check data
    data_dir = _REPO_ROOT / "current" / "penny_research" / "data"
    bar_cache = data_dir / "bar_cache"
    survey = data_dir / "phase3_full.parquet"

    diag["data"] = {
        "bar_cache_exists": bar_cache.exists(),
        "bar_cache_files": len(list(bar_cache.glob("*.parquet"))) if bar_cache.exists() else 0,
        "survey_exists": survey.exists(),
        "survey_size_kb": round(survey.stat().st_size / 1024, 1) if survey.exists() else 0,
    }

    # Check imports
    try:
        from penny_genes import sample_candidates, ENTRY_TYPE_CATALOG
        diag["genes"] = {
            "entry_types": len(ENTRY_TYPE_CATALOG),
            "sample_ok": True,
        }
    except Exception as e:
        diag["genes"] = {"error": str(e)}

    try:
        from penny_search import PROFILES
        diag["profiles"] = list(PROFILES.keys())
    except Exception as e:
        diag["profiles"] = {"error": str(e)}

    diag["environment"] = {
        "python_version": sys.version,
        "cpu_count": mp.cpu_count(),
        "PENNY_WORKERS": os.environ.get("PENNY_WORKERS", ""),
    }

    return diag


# ─── RunPod entry point ─────────────────────────────────────────────────────

import runpod

runpod.serverless.start({"handler": handler})

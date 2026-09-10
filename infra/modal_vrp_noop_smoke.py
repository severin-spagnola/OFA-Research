#!/usr/bin/env python3
"""
Noop smoke sentinel for OFA-VRP.

Verifies:
  1. Container image builds successfully
  2. /root/options_vrp is on sys.path
  3. wf_runner imports (evaluate_window, WFWindow, KillConfig) succeed inside the container

No actual compute is performed.

Usage:
    modal run infra/modal_vrp_noop_smoke.py
"""
import json
import sys
from pathlib import Path

import modal

app = modal.App("ofa-vrp-noop-smoke")

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


@app.function(image=image, timeout=120)
def noop_window() -> dict:
    """Validate container sys.path and wf_runner imports; return sentinel dict."""
    sys.path.insert(0, "/root/options_vrp")
    from wf_runner import evaluate_window, WFWindow, KillConfig  # noqa: F401
    return {"window_id": "smoke_sentinel", "smoke": True, "ok": True}


@app.local_entrypoint()
def main():
    sentinel = noop_window.remote()
    _repo_root = Path(__file__).resolve().parent.parent
    out_path = (
        _repo_root
        / "current"
        / "options_vrp"
        / "results"
        / "options_vrp"
        / "regime_db.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(sentinel) + "\n")
    assert sentinel.get("ok") is True, f"noop_window returned unexpected sentinel: {sentinel}"
    print("SMOKE PASS")
    sys.exit(0)

#!/usr/bin/env python3
"""
VRP Walk-Forward Iter3 Dispatch
================================
Submits iter3 walk-forward job batch to RunPod endpoint eaxyxd1j4xxcib.

Iter3 params:
  - 400 candidates per window
  - seed 456 (+ window_id * 137 per window)
  - max-concurrent 5
  - output: results/options_vrp/wf_results_iter3.jsonl

Orchestrator auto-discovers dates via diagnostic job, then submits one
vrp_walkforward job per window and polls until all complete, appending
regime records to the output file.

Usage:
    python infra/vrp_wf_iter3_dispatch.py
    python infra/vrp_wf_iter3_dispatch.py --start-window 10   # resume
    python infra/vrp_wf_iter3_dispatch.py --diagnostic        # check endpoint
    python infra/vrp_wf_iter3_dispatch.py --overwrite         # re-run all
"""
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv('/Users/severinspagnola/Desktop/Strat/.env')
if not os.environ.get("RUNPOD_API_KEY"):
    sys.exit("ERROR: RUNPOD_API_KEY not set in /Users/severinspagnola/Desktop/Strat/.env")

# Inject iter3 defaults before wf_orchestrator parses sys.argv
_ITER3_DEFAULTS = [
    "--endpoint",       os.environ.get('VRP_ITER3_ENDPOINT_ID', 'eaxyxd1j4xxcib'),  # VERIFIED 2026-03-21: endpoint ID confirmed correct (matches step 1 discovery; no change needed)
    "--candidates",     "400",
    "--seed",           "456",
    "--max-concurrent", "5",
    "--output",         "results/options_vrp/wf_results_iter3.jsonl",
]

# Prepend defaults — user-supplied flags will override them since argparse
# uses last-wins for duplicate flags when parsed left-to-right is not the
# case; instead, we inject only flags NOT already present in sys.argv.
_existing_flags = set(a for a in sys.argv[1:] if a.startswith("--"))
_inject = []
it = iter(_ITER3_DEFAULTS)
for flag, value in zip(it, it):
    if flag not in _existing_flags:
        _inject += [flag, value]

sys.argv[1:1] = _inject  # insert before any user-supplied args

# Resolve repo root so imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "infra"))

# Import and run the orchestrator
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "wf_orchestrator",
    _REPO_ROOT / "infra" / "wf_orchestrator.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.main()

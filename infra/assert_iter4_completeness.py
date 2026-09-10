import json
import pathlib
import sys

JSONL_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "current" / "options_vrp" / "results" / "options_vrp" / "wf_results_iter4.jsonl")
N_EXPECTED = 36

with open(JSONL_PATH) as f:
    lines = [json.loads(line) for line in f if line.strip()]

filtered = [r for r in lines if r.get("window_id") != "smoke_sentinel"]
n = len(filtered)

if n >= N_EXPECTED:
    print(f"ok N_EXPECTED={N_EXPECTED} got={n}")
else:
    print(f"FAIL: expected >= {N_EXPECTED} windows, got {n}", file=sys.stderr)
    sys.exit(1)

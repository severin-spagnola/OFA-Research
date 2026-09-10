import json, pathlib, sys
JSONL_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "current" / "options_vrp" / "results" / "options_vrp" / "regime_db.jsonl")
lines = [json.loads(l) for l in open(JSONL_PATH) if l.strip()]
filtered = [r for r in lines if r.get("window_id") != "smoke_sentinel"]
n = len(filtered)
assert n >= 1, f"FAIL: expected >= 1 non-sentinel regime_db records, got {n}"
print(f"ok regime_db got={n}")

import json
import sys
from pathlib import Path

import modal

app = modal.App("ofa-smoke-minimal")

SMOKE_PATH = Path(__file__).parent.parent / "current/options_vrp/results/options_vrp/smoke_test.jsonl"


@app.function()
def noop_window():
    return {"window_id": 0, "smoke": True, "source": "modal_smoke_minimal"}


@app.local_entrypoint()
def dispatch():
    record = noop_window.remote()

    SMOKE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SMOKE_PATH, "w") as f:
        f.write(json.dumps(record) + "\n")

    lines = [l for l in SMOKE_PATH.read_text().splitlines() if l.strip()]
    if len(lines) == 1:
        print("smoke_ok=1")
    else:
        sys.exit(1)

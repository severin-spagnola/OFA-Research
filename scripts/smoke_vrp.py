#!/usr/bin/env python3
"""Smoke test runner for modal_vrp — invokes modal run and validates the sentinel file."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SENTINEL_PATH = REPO_ROOT / "current" / "options_vrp" / "results" / "options_vrp" / "smoke_test.jsonl"


def main():
    # Step 1: invoke modal run, streaming stdout/stderr live
    cmd = ["modal", "run", "infra/modal_vrp.py", "--", "--smoke", "True"]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=False,  # stream live to terminal
    )

    # Step 2: assert exit code == 0
    if result.returncode != 0:
        print(f"\nSMOKE FAIL: modal run exited {result.returncode}")
        sys.exit(1)

    # Step 3/4: resolve and assert sentinel file exists
    sentinel = SENTINEL_PATH
    if not sentinel.exists():
        print(f"SMOKE FAIL: sentinel file not found at {sentinel}")
        sys.exit(1)

    # Step 5/6: read and validate every non-empty line as JSON
    lines = sentinel.read_text().splitlines()
    valid_lines = []
    for i, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            json.loads(raw)
            valid_lines.append(raw)
        except json.JSONDecodeError:
            print(f"SMOKE FAIL: invalid JSON on line {i}: {raw}")
            sys.exit(1)

    if not valid_lines:
        print("SMOKE FAIL: sentinel file is empty")
        sys.exit(1)

    # Step 7: success
    print(f"SMOKE PASS — {len(valid_lines)} sentinel lines validated")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
validate_smoke_test.py

Reads current/options_vrp/results/options_vrp/smoke_test.jsonl and validates
that it contains real wf_runner output records (not smoke stubs).

Assertions:
  (a) at least 1 non-empty JSON line exists
  (b) every record has a 'window_id' integer field
  (c) every record has at least one of 'n_trades', 'pnl', or 'regime'

Prints each record and ends with '[validate_smoke_test] PASSED' on success.
Exits 1 on any failure.
"""

import json
import sys
from pathlib import Path

JSONL_PATH = (
    Path(__file__).parent.parent
    / "current"
    / "options_vrp"
    / "results"
    / "options_vrp"
    / "smoke_test.jsonl"
)


def main() -> None:
    path = JSONL_PATH.resolve()
    if not path.exists():
        raise FileNotFoundError(f"smoke_test.jsonl not found: {path}")

    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"JSON parse error on line {line_no}: {exc}"
                ) from exc
            records.append(rec)

    # ── (a) At least 1 non-empty JSON line ────────────────────────────────────
    assert len(records) >= 1, "No non-empty JSON records found in smoke_test.jsonl"

    for i, rec in enumerate(records):
        print(f"[validate_smoke_test] record[{i}]: {rec}")

        # ── (b) Every record has a 'window_id' integer field ──────────────────
        assert "window_id" in rec, (
            f"Record {i}: missing 'window_id' field. Full record: {rec}"
        )
        assert isinstance(rec["window_id"], int), (
            f"Record {i}: 'window_id' is not an integer (got {type(rec['window_id']).__name__}). "
            f"Full record: {rec}"
        )

        # ── (c) Every record has at least one of n_trades, pnl, or regime ─────
        real_keys = {"n_trades", "pnl", "regime"}
        assert real_keys & rec.keys(), (
            f"Record {i}: missing all of 'n_trades', 'pnl', 'regime' — not a real wf_runner record. "
            f"Full record: {rec}"
        )

    print("[validate_smoke_test] PASSED")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError) as exc:
        print(f"[validate_smoke_test] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)

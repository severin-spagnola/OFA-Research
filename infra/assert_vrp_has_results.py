#!/usr/bin/env python3
"""
Assert regime_db.jsonl contains at least one non-smoke result row.

Checks:
(a) File exists.
(b) At least one non-empty line that is valid JSON.
(c) At least one record has a window_id that is a non-smoke integer
    (i.e. not the string 'smoke_sentinel').

Prints 'ASSERT OK' and exits 0 on pass; prints failure reason and exits 1 on fail.
"""
import argparse
import json
import sys
from pathlib import Path


def fail(reason: str) -> None:
    print(f"ASSERT FAIL: {reason}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assert regime_db.jsonl has at least one real (non-smoke) result row."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("current/options_vrp/results/options_vrp/regime_db.jsonl"),
        help="Path to the JSONL file to validate.",
    )
    args = parser.parse_args()

    path: Path = args.file

    # (a) file exists
    if not path.exists():
        fail(f"File not found: {path}")

    # (b) at least one non-empty line that is valid JSON
    records = []
    for i, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append((i, json.loads(line)))
        except json.JSONDecodeError as exc:
            fail(f"JSON parse error on line {i}: {exc}")

    if not records:
        fail("File contains no non-empty lines")

    # (c) at least one record has a window_id that is a non-smoke integer
    non_smoke = [
        (lineno, r)
        for lineno, r in records
        if "window_id" in r and r["window_id"] != "smoke_sentinel" and isinstance(r["window_id"], int)
    ]

    if not non_smoke:
        fail(
            f"No record with a non-smoke integer window_id found "
            f"({len(records)} total record(s), all are smoke or missing window_id)"
        )

    print(f"ASSERT OK: {len(non_smoke)} non-smoke result row(s) found in {path}")
    sys.exit(0)


if __name__ == "__main__":
    main()

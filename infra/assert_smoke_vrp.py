#!/usr/bin/env python3
"""
Assert correctness of options_vrp smoke output.

Checks:
(a) File exists.
(b) At least 1 non-empty line.
(c) Last record parses as JSON and contains 'smoke': True.

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
    parser = argparse.ArgumentParser(description="Assert correctness of options_vrp smoke output.")
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

    # (b) at least 1 non-empty line
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if len(lines) == 0:
        fail("File contains no non-empty lines")

    # (c) last record parses as JSON and contains 'smoke': True
    last_line = lines[-1]
    try:
        record = json.loads(last_line)
    except json.JSONDecodeError as exc:
        fail(f"JSON parse error on last record: {exc}")

    if record.get("smoke") is not True:
        fail(f"smoke={record.get('smoke')!r} in last record, expected True")

    print("ASSERT OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

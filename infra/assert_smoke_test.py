#!/usr/bin/env python3
"""
Assert correctness of modal_smoke_minimal output.

Checks:
(a) File exists and contains exactly 1 non-empty line.
(b) That line is valid JSON.
(c) window_id == 0
(d) smoke == true
(e) source == 'modal_smoke_minimal'

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
    parser = argparse.ArgumentParser(description="Assert correctness of modal_smoke_minimal output.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("current/options_vrp/results/options_vrp/smoke_test.jsonl"),
        help="Path to the smoke_test JSONL file to validate.",
    )
    args = parser.parse_args()

    path: Path = args.file

    if not path.exists():
        fail(f"File not found: {path}")

    lines = [l for l in path.read_text().splitlines() if l.strip()]

    # (a) exactly 1 non-empty line
    if len(lines) != 1:
        fail(f"Expected exactly 1 record, got {len(lines)}")

    # (b) valid JSON
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        fail(f"JSON parse error: {exc}")

    # (c) window_id == 0
    if record.get("window_id") != 0:
        fail(f"window_id={record.get('window_id')!r}, expected 0")

    # (d) smoke == true
    if record.get("smoke") is not True:
        fail(f"smoke={record.get('smoke')!r}, expected true")

    # (e) source == 'modal_smoke_minimal'
    if record.get("source") != "modal_smoke_minimal":
        fail(f"source={record.get('source')!r}, expected 'modal_smoke_minimal'")

    print("ASSERT OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

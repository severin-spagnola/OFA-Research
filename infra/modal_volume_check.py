#!/usr/bin/env python3
"""
List the contents of the 'ofa-vrp-data' Modal volume at /options_5dte.

Volume-internal path: /options_5dte  (use: modal volume ls ofa-vrp-data /options_5dte)
Container mount:      /root/data  → /root/data/options_5dte inside the function

Usage:
    modal run infra/modal_volume_check.py
"""
import modal

app = modal.App("ofa-vrp-volume-check")

data_vol = modal.Volume.from_name("ofa-vrp-data", create_if_missing=False)


@app.function(
    volumes={"/root/data": data_vol},
    timeout=60,
)
def list_chain_data(path: str = "/root/data/options_5dte") -> dict:
    import os
    from pathlib import Path

    target = Path(path)
    result = {"path": path, "exists": target.exists(), "is_dir": False, "entries": []}

    if not target.exists():
        return result

    result["is_dir"] = target.is_dir()
    if result["is_dir"]:
        entries = sorted(os.listdir(target))
        result["entries"] = entries
        result["count"] = len(entries)

    return result


@app.function(
    volumes={"/root/data": data_vol},
    timeout=60,
)
def walk_volume_root(root: str = "/root/data", max_depth: int = 3) -> None:
    import os

    root = root.rstrip("/")
    root_depth = root.count(os.sep)

    print(f"[WALK] {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        current_depth = dirpath.count(os.sep) - root_depth
        if current_depth >= max_depth:
            dirnames.clear()
            continue
        indent = "  " * current_depth
        print(f"{indent}[DIR]  {dirpath}/")
        for fname in sorted(filenames):
            print(f"{indent}  {fname}")
        dirnames.sort()


@app.local_entrypoint()
def main():
    walk_volume_root.remote()
    result = list_chain_data.remote()
    path = result["path"]
    if not result["exists"]:
        print(f"[FAIL] Path does not exist: {path}")
        return
    if not result["is_dir"]:
        print(f"[FAIL] Path exists but is not a directory: {path}")
        return

    count = result.get("count", 0)
    entries = result.get("entries", [])
    print(f"[OK] {path} — {count} entries")
    for e in entries:
        print(f"  {e}")

#!/usr/bin/env python3
"""
analyze_vrp_iter3.py
- Reads regime_db.jsonl
- Strips smoke_sentinel records
- Annotates each of the 28 window records with 'winner' bool
  (winner = oos_auc >= 0.93 AND fitness >= 1.0)
- Overwrites regime_db.jsonl in-place with the cleaned 28-record file
- Prints a summary table
"""

import json
import pathlib

REGIME_DB = pathlib.Path(__file__).parent.parent / (
    "current/options_vrp/results/options_vrp/regime_db.jsonl"
)

OOS_AUC_THRESHOLD = 0.93
FITNESS_THRESHOLD = 1.0


def main():
    records = []
    with REGIME_DB.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    # Strip smoke sentinels
    windows = [r for r in records if r.get("window_id") != "smoke_sentinel"]
    skipped = len(records) - len(windows)
    print(f"Loaded {len(records)} records — stripped {skipped} smoke_sentinel(s), keeping {len(windows)} windows.")

    # Annotate winner
    for w in windows:
        w["winner"] = bool(
            w.get("oos_auc", 0.0) >= OOS_AUC_THRESHOLD
            and w.get("fitness", 0.0) >= FITNESS_THRESHOLD
        )

    # Overwrite in-place
    with REGIME_DB.open("w") as f:
        for w in windows:
            f.write(json.dumps(w) + "\n")

    print(f"Overwrote {REGIME_DB} with {len(windows)} records.")

    # Summary
    winners = [w for w in windows if w["winner"]]
    print()
    print(f"{'='*60}")
    print(f"  Total windows : {len(windows)}")
    print(f"  Winners       : {len(winners)}  (oos_auc >= {OOS_AUC_THRESHOLD} AND fitness >= {FITNESS_THRESHOLD})")
    print(f"{'='*60}")

    if winners:
        print(f"  {'window_id':>10}  {'oos_auc':>9}  {'fitness':>9}")
        print(f"  {'-'*10}  {'-'*9}  {'-'*9}")
        for w in winners:
            print(f"  {str(w['window_id']):>10}  {w['oos_auc']:>9.4f}  {w['fitness']:>9.5f}")
    else:
        print("  No winners found.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit a FitRank submission CSV."""

import csv
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/audit_shortlist.py outputs/submission.csv")
        return 1
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"Rows: {len(rows)}")
    titles = Counter(row["reasoning"].split(" | ")[0] for row in rows)
    print("Title distribution:")
    for title, count in titles.most_common(10):
        print(f"  {title}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

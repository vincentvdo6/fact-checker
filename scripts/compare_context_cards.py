"""Compare two saved clicks against fixed contextual preservation expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.context_cards import compare_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("expectations", type=Path)
    args = parser.parse_args()
    try:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in
                   (args.baseline, args.candidate, args.expectations)]
        report = compare_records(*records)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {"ok": False, "error": str(error)}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

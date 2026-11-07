"""Print the evaluation results table. Track B, owner: babarchechi. See docs/PHASE1_PLAN.md, Section 11.

1. Record each run:   python src/main.py --source 0 --scenario 6
2. Then, from the repo root:   python src/eval_report.py
3. Paste the output into docs/evaluation.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sentra.evaluation import SCENARIOS, load_sessions, markdown_table, score, summary
from sentra.pipeline import DEFAULT_OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Score evaluation runs recorded with main.py --scenario N")
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder with transitions_*.csv files")
    args = parser.parse_args()

    sessions, unlabelled = load_sessions(args.dir)
    if not sessions:
        print(f"No evaluation runs found in {args.dir}/.")
        print("Record each run with:  python src/main.py --source 0 --scenario <1-9>")
        for number, scenario in SCENARIOS.items():
            print(f"  {number}: {scenario.description}")
        return

    results = score(sessions)
    print(markdown_table(results))
    print()
    print(summary(results))
    if unlabelled:
        print(f"\n(Skipped {len(unlabelled)} run(s) recorded without --scenario.)")


if __name__ == "__main__":
    main()

"""Run with python -m research.study; no import-time computations."""

import argparse
import json

from .data import ROOT, load_data, monthly_targets, sources
from .report import render
from .runner import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    run = sub.add_parser("run")
    run.add_argument(
        "--profile", choices=["smoke", "core", "sensitivity", "external"], default="smoke"
    )
    run.add_argument("--resume")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--hours", type=float, default=4)
    report = sub.add_parser("report")
    report.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        data = load_data()
        import pandas as pd

        original = pd.read_csv(
            ROOT / "data/merged_macroeconomic_credit.csv", index_col="Month_End", parse_dates=True
        )
        values, _ = sources()
        rebuilt = monthly_targets(values["BAMLH0A0HYM2"])
        error = (rebuilt.mean_legacy_ffill - original.Credit_Spread).abs().max()
        if error > 1e-10:
            raise ValueError("Legacy target fails reconciliation")
        print(json.dumps(data.quality | {"legacy_reconciliation_max_error": error}, indent=2))
    elif args.command == "run":
        execute(args.profile, args.resume, args.workers, args.hours)
    else:
        render(args.run)


if __name__ == "__main__":
    main()

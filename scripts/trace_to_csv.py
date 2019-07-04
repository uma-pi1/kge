#!/usr/bin/env python
from kge.job import Trace
import argparse
import sys
import yaml

# Example:
# grep -e "job: train.*scope: epoch" local/experiments/toy/trace.yaml \
# | trace_to_csv.py epoch timestamp avg_loss
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert trace file to CSV (reads stdin, writes stdout)"
    )
    parser.add_argument("fields", type=str, nargs="*", help="field names to include")
    args = parser.parse_args()

    trace = Trace()
    for line in sys.stdin:
        trace.entries.append(yaml.load(line, Loader=yaml.SafeLoader))

    df = trace.to_dataframe()
    if args.fields:
        df = df[args.fields]
    df.to_csv(sys.stdout, index=False)

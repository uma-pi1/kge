import argparse
import sys
import yaml

# Example:
# grep -e "job: train.*scope: epoch" local/experiments/toy/trace.yaml \
# | python trace_to_csv.py epoch timestamp avg_loss
# or:
# grep -e "job: eval.*scope: epoch" local/experiments/toy/trace.yaml \
# | python trace_to_csv.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert trace file to CSV (reads stdin, writes stdout)")
    parser.add_argument('fields', type=str, nargs='*',
                        help="field names to include")
    args = parser.parse_args()
    if len(args.fields) > 0:
        print(",".join(args.fields))  # header line
    for line in sys.stdin:
        kv_pairs = yaml.load(line, Loader=yaml.SafeLoader)
        if len(args.fields) == 0:
            args.fields = list(kv_pairs.keys())
            print(",".join(args.fields))  # header line
        sep = ''
        for field in args.fields:
            print(sep, end='')
            print(kv_pairs[field], end='')
            sep = ','
        print()

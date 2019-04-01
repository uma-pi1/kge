import argparse
import sys
import yaml

# Example:
# grep -e "type: train_epoch" local/experiments/toy/trace.yaml \
# | python trace_to_csv.py epoch timestamp avg_loss
# or:
# grep -e "type: eval_er_epoch" local/experiments/toy/trace.yaml \
# | python trace_to_csv.py epoch timestamp mean_reciprocal_rank mean_reciprocal_rank_filtered
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert trace file to CSV (reads stdin, writes stdout)")
    parser.add_argument('fields', type=str, nargs='+',
                        help="field names to include")
    args = parser.parse_args()

    print(",".join(args.fields))  # header line
    for line in sys.stdin:
        kv_pairs = yaml.load(line, Loader=yaml.SafeLoader)
        sep = ''
        for field in args.fields:
            print(sep, end='')
            print(kv_pairs[field], end='')
            sep = ','
        print()

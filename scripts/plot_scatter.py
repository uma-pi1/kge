#!/usr/bin/env python
#
# Scatter plot of specified fields/expressions of trace entries (read from stdin). In
# expressions, dots need to be quoted (\\.) Arguments are: x, y, and optionally groupby
# expression.
#
# Example: plot_scatter.py plot.pdf "num_active_parameters.fillna(num_parameters)" mean_reciprocal_rank_filtered_with_test model
from kge.job import Trace
import pandas as pd
import yaml
import sys
import matplotlib.pyplot as plt

trace = Trace()
for line in sys.stdin:
    entry = yaml.load(line, Loader=yaml.SafeLoader)
    trace.entries.append(entry)

filename = sys.argv[1]
x_expr = sys.argv[2]
y_expr = sys.argv[3]
if len(sys.argv) > 4:
    groupby_expr = sys.argv[4]
else:
    groupby_expr = None
df = trace.to_dataframe()
df.columns = list(map(lambda s: s.replace(".", "___"), df.columns))
x = eval(x_expr.replace("\\.", "___"), None, df)
y = eval(y_expr.replace("\\.", "___"), None, df)
if groupby_expr:
    groupby = eval(groupby_expr.replace("\\.", "___"), None, df)
    print(pd.concat([x, y, groupby], axis=1))
else:
    groupby = None
    print(pd.concat([x, y], axis=1))

print("Plotting to {}".format(filename))
fig = plt.figure()
if groupby is None:
    plt.scatter(x, y)
else:
    for group in groupby.unique():
        mask = groupby == group
        plt.scatter(x[mask], y[mask], label=group, marker="x")
    plt.legend()
plt.xlabel(x_expr)
plt.ylabel(y_expr)
fig.savefig(filename, bbox_inches="tight")

from kge.job import Trace
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interpol
import numpy as np

"""
Program to plot several trace files.
The metric value is plotted against the time spent in the iteration.
The mean metric of the iterations of each algorithm are calculated as well as their sleeves.
The sleeves just represent the min loss and the max loss.

Please adjust the following parameters to suit to your needs:
"""
# The trace file lies in the directory: base_url/algorithm-dataset-train_type
dataset = 'wnrr'
train_type = 'neg-samp'

algorithms = ['ax', 'bohb', 'hpb', 'rnd', 'tpe']
algo_names = ['Ax', 'BOHB', 'Hyperband', 'Random', 'TPE']
line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

base_url = "/home/felix/PycharmProjects/kge/local/experiments/"

output = dataset + "-" + train_type
x_expr = "Wall clock time [s]"
y_expr = "metric_value"

########################################################################################
fig = plt.figure()
for a in range(len(algorithms)):
    directory = algorithms[a] + "-" + dataset + "-" + train_type
    trace_url = base_url + directory + "/trace.yaml"

    trace = Trace(trace_url)
    df = trace.to_dataframe()

    # Delete first row (run information) and lsat row (best result)
    df = df[1:df.shape[0]-1]
    # Folder is equals the iteration
    df = df[['folder', 'timestamp', 'metric_value']]
    df.folder = pd.to_numeric(df.folder, errors='coerce')
    df.dropna(inplace=True)

    # Calculate wall clock times of each iteration
    # Smooth the time values by factor X=500
    group = df['timestamp'].groupby(df['folder']).min()
    df['time'] = 0
    for i in df.index:
        df['time'].loc[i] = round((df['timestamp'].loc[i] - group.loc[df['folder'].loc[i]]) / 500) * 500

    df2 = df['metric_value'].groupby(df['time']).describe()

    xnew = np.linspace(df2.index.min(), df2.index.max(), 1000)

    cs_mean = interpol.CubicSpline(df2.index, df2['mean'])
    cs_min = interpol.CubicSpline(df2.index, df2['min'])
    cs_max = interpol.CubicSpline(df2.index, df2['max'])

    plt.plot(xnew, cs_mean(xnew), label=algo_names[a], color=line_colors[a])
    plt.fill_between(xnew, cs_min(xnew), cs_max(xnew), color=line_colors[a], alpha=0.1)

    print("Done plotting {}".format(algo_names[a]))

plt.legend()
plt.xlabel(x_expr)
plt.ylabel(y_expr)
plt.show()

print("Plotting to {}".format(output))
fig.savefig(output, bbox_inches="tight")
print("Finished")
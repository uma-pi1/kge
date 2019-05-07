# examples for analyzing the output of a search job

import matplotlib.pyplot as plt
from IPython import get_ipython
from kge.job import Trace

get_ipython().magic("matplotlib")

# put tracefile and job id of your search job here
tracefile = "/home/rgemulla/extern/kge/local/experiments/20190506-182257-fb15k-237-complex/trace.yaml"
job_id = "a1f0efad"

# load the trace (i.e., the part completed by now)
trace = Trace(tracefile, job_id)

# extract all data that is associated to entire training jobs
train = trace.to_dataframe({"scope": "train"})
print(train.columns)
print(len(train))

# the last result
print(train[-1:].transpose())

# scatter plot of all results so far as a function of the learning rate
plt.clf()
plt.scatter(train["train.optimizer_args.lr"], train.metric_value)

# lood detail data (again, just the part being completed)
epoch = trace.to_dataframe({"scope": "epoch"})

# scatter plot epoch vs metric
plt.clf()
plt.scatter(epoch["epoch"], epoch.metric_value)

# here again (TODO legend should list folder names)
plt.clf()
epoch.groupby("folder").plot("epoch", "metric_value", ax=plt.gca())

# sandbox
tracefile = "/home/rgemulla/extern/kge/local/experiments/fb15k-237-complex-search-large/trace.yaml"
job_id = "9b2846dd"

# examples for analyzing the output of a search job

from kge.job import Trace
import matplotlib.pyplot as plt
from IPython import get_ipython

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
print(train.iloc[-1,:].transpose())

# the best result
print(train.iloc[train["metric_value"].idxmax(),:].transpose())

# scatter plot of all results so far as a function of the learning rate
plt.clf()
plt.scatter(train["train.optimizer_args.lr"], train.metric_value)

# scatter plot of all results so far as a function of time
plt.clf()
plt.scatter(train["timestamp"] - min(train["timestamp"]), train.metric_value)
plt.xlabel("Seconds")
plt.ylabel(train["metric_name"][0])

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

tracefile = (
    "/home/rgemulla/extern/kge/local/experiments/fb15k-237-complex-axsearch/trace.yaml"
)
job_id = "0f3d6784"

tracefile = "/home/rgemulla/extern/kge/local/experiments/fb15k-237-complex-axsearch50/trace.yaml"
job_id = "c4bac385"

# libKGE: A library for Knowledge Graph Embeddings

libKGE is a library for training, evaluating and tuning [knowledge graph
embeddings](https://ieeexplore.ieee.org/document/8047276) (KGE). It is
based on [PyTorch](https://pytorch.org/) and designed to be easy to use
and easy to extend. libKGE is highly flexible for training and tuning KGE
models, as it supports various combinations of loss functions, optimizers,
training types and many more hyperparameters. Hyperparameter optimization
is also supported in different ways, e.g. grid search, pseudo-random search
or Bayesian optimization. These are some of the state-of-the-art results
obtained with libKGE (mean reciprocal rank):

<center>

|          | FB15K-237 | WNRR |
|----------|-----------|------|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)   | 0.36      | 0.46 |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)   | 0.31      | 0.42 |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf) | 0.35      | 0.45 |  
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)  | 0.35      | 0.47 |
| [ConvE](https://arxiv.org/abs/1707.01476)    | 0.34      | 0.44 |

</center>



# Quick start

libKGE supports training, evaluating and tuning KGE models. The settings for
each task can be specified with a configuration file in YAML format.
There are many available settings, all of which have default values and
can be found in [config-default.yaml](kge/config-default.yaml).

## Training a model

To train a model, define a configuration file, for example:

```yaml
job.type: train
dataset.name: fb15k-237
model: complex
train:
  optimizer: Adagrad
  optimizer_args:
    lr: 0.2
lookup_embedder.dim: 100
lookup_embedder.regularize_weight: 0.8e-7
valid:
  every: 5
  metric: mean_reciprocal_rank_filtered

# All non-specified settings take their default value from config-default.yaml
```
To begin training, run one of the following:

```
# The start command creates an output folder and begins training
# This creates a copy of the config file in the output folder
python kge.py start config.yaml

# The create command creates an output folder but does not begin training
python kge.py create config.yaml

# To begin training on an existing folder, run the following in that folder:
python kge.py resume .

# You may specify the output folder and device
python kge.py start config.yaml --folder kge_test --job.device cuda:0

# All entries in the config file can be overwritten in the command line, e.g.
python kge.py start config.yaml --train.optimizer Adam
```

## Recovering from an interruption

All tasks can be resumed if interrupted. Run the following in the
corresponding output folder:

```
python kge.py resume .

# Change the device when resuming
python kge.py resume kge_test/config.yaml --job.device cuda:1

```

## Evaluating a model

To evaluate trained model, run the following:

```
# Evaluate a model on the validation split
python kge.py valid kge_test/config.yaml

# Evaluate a model on the test split
python kge.py test kge_test/config.yaml
```

## Tuning a model

libKGE supports various forms of hyperparameter optimization. e.g. grid
search or Bayesian optimization. The search type and search space are
specified in the configuration file. We use [Ax](https://ax.dev/) for
SOBOL (pseudo-random) and Bayesian optimization. For example, the
following config file defines a search of 10 SOBOL trials (arms)
followed by 10 Bayesian optimization trials:

```yaml
job.type: search
search.type: ax
dataset.name: wnrr
valid.metric: mean_reciprocal_rank_filtered
model: reciprocal_relations_model
reciprocal_relations_model.base_model.type: conve
ax_search:
  num_trials: 30
  num_sobol_trials: 10  # remaining trials are Bayesian
  parameters:
    - name: train.batch_size
      type: choice   
      values: [256, 512, 1024]
    - name: train.optimizer_args.lr     
      type: range
      bounds: [0.0003, 1.0]
    - name: train.type
      type: fixed
      value: 1vsAll
```
Trials can be run in parallel across several devices (evenly distributed
across available devices):

```
# Run 4 trials in parallel across two GPUs
python kge.py resume . --search.device_pool cuda:0,cuda:1 --search.num_workers 4
```

## Other commands
To see all available commands:

```
python kge.py --help
```

# Installation

To install libKGE, clone this repository and install the requirements with pip:


```
git clone https://github.com/uma-pi1/kge.git
pip install -e .
```

# Supported KGE models

libKGE has implementations for the following KGE models:

- [RESCAL](kge/model/rescal.py)
- [Transe](kge/model/transe.py)
- [DistMult](kge/model/distmult.py)
- [ComplEx](kge/model/complex.py)
- [Conve](kge/model/conve.py)

The [examples](examples) folder contains some configuration files as examples
of how to run these models.

# Adding a new model

To add a new model to libKGE, one needs to extend the
[KgeModel](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L243)
class. A model is made up of a
[KgeEmbedder](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L170)
to associate each subject, relation and object to an embedding, and a
[KgeScorer](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L76)
to score triples.

# Guidelines
- Do not add any datasets or experimental code to this repository
- Code formatting using [black](https://github.com/ambv/black) and with default
  settings (line length 88)
- Code documentation following [Google Python Style Guide (Sec.
  3.8)](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings);
  see
  [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Use (type
  annotations)[https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html]
  whenever considered helpful
  - For tensors, use `torch.Tensor` only (for all tensors/shapes/devices); do
    not use any more specific annotations (e.g., `torch.LongTensor` refers to a
    CPU tensor only, but not a CUDA tensor)
- Unspecified configuration values are indicated by
  - `''` for strings
  - `-1` for non-negative integers
  - `.nan` for floats
  
  
# Visualization Support
The framework supports [visdom](https://github.com/facebookresearch/visdom) and  [tensorboard](https://www.tensorflow.org/tensorboard)
as visualization engines to plot data that was produced while executing jobs. For further infos and all options please also check the config
key `visualize` in config_default.yaml.
## Installation 
For installing the visualization support, tensorboard and/or visdom is needed. 
- run locally `pip install -e .[visualize]`
- alternatively, install visdom or tensorboard separately

## Modes
While training a model or running a search job, the training loss, the `valid.metric` and various other
quantities produced can be **broadcasted** simultaneously (visdom only). Alternatively, trace files can be
**post**-processed and plots can be generated after job execution.<br>**Note**: Broadcasting is performed during job execution and therefore can directly
 affect training times whereas post-processing sessions are independent of job execution. <br>Finally, data that is broadcasted can directly be compared with
data from older executions. 

## General Options
With the options `visualize.include_eval`, `visualize.include_train` metrics can be selected which will be plotted against the epoch in a visualization session. Leaving these lists empty
will lead to all metrics being selected but this can lead to performance instabilites.
On the other hand, `visualize.exclude_eval`, `visualize.exclude_train` can be used to de-select specific metrics.   
## Post-Processing
Post-Processing sessions take a specific config file and are started with the command `visualize`. Please see config-default.yaml for all options. Run `python kge.py visualize examples/visualize-options.yaml` to start a visualization session.
In `visualize.post.search_dir` a relative directory where the program will search for job folders can be specified. 
In  `visualize.post.folders` folder names or regex patterns can be entered to decide which folders to select.
If `visualize.post.folders` specifies an empy list, then all folders will be selected.

### Visdom Post-Processing
After running the the post processing command `python kge.py visualize examples/visualize-options.yaml` 
with module: visdom, the session can be opened in the browser at the specified host and port.<br><br>
You will see a selection field at the top of the page where all folder names are listed. These so called 'environments' can 
be selected by clicking on the respective checkbox.<br> Training jobs only have one checkbox whereas search jobs follow a nested structure 
and also have a summary environment. Click on one of the checkboxes to open the environment. This will show a window which displays the config of the 
respective job and a window with a `syncronize` button. Click this button to load data into the environment. <br>

For a training job environment, when `synchronize` is clicked, all the specified metrics plotted against the epoch number will be loaded.
**Tipp**: To compare data between two or multiple training jobs, synchronize them and then select their checkboxes simultaneously. This will combine all line charts
with the same title together in single line plots.

For a search job, after clicking `syncronize` sub environments are created which contain data of the search trials (sub training jobs). 

Additionally, for a search job, when you click `synchronize` in the {folder}_SummaryEnvironment you will see the following windows:
- A bar plot window with title {valid.metric}_best which shows for every search trial its best result for the valid.metric. **Tipp:** you can move the mouse into the window and it will directly show you on top
the value of the best job. In the window options on the top of the window you can also select "show closest data on hover" to parse through
the results of single trials.<br><br>
- A line chart all_{valid.metric} where every search trial's progress of {valid.metric} against the epoch number is plotted. **Tipp**: Click on one of the names in the legend to de-select one of the lines. Double click on one of the names in the legend to exclusively select one of the lines. <br><br>
- Scatter plots that show the best {valid.metric} against the values of the hyperparmeters which were assigned to  the search trials by the search algorithm.<br><br>
- Bar plots which show the values of the hyperparamters for every search trial. Every bar denotes the value that was assigned to the respective trial.  

### Tensorboard Post-Processing
Activate tensorboard by setting the option`visualize.module: tensorboard` then run the visualize command as above. It is suggested to only add some specific folders
in the `visualize.post.folders` option such that the tensorboard interface is not overcrowded. When using tensorboard, the option `visualize.post.embeddings` can be
used to show the embeddings of the respective model with the tensorboard projector functionality.

   
## Visdom Broadcasting
Broadcasting is supported only for module:visom. When running a training job or a search job use the flag `visualize.broadcast.enable: True` this will start the server
at the specified port and host and the progress of the executed job can be monitored synchronously.
Note that the config file to run the kge job is directly used to also specify the visualization options. For instance, define the options
`visualize.include_eval` and `visualize.include_train` in your usual config file to define which metrics shall be tracked in broadcasting.

### Comparing a job that is currently broadcasted with older data
In general, visdom will always restart the server when a new session is started. If you want to compare a job which is currently run with older jobs
then start a post-processing session and syncronize the jobs you want to compare. Then start your kge job with the `visualize.broadcast.enable` option.
Additionally set the option `visualize.start_server: False` to not restart the server. This will append your currently broadcasted job to the post-processing
session. And by selecting multiple environments simultaneously, you can directly compare them. Note: this can lead to a performance decrease and should only be 
done for investigating/debugging model configurations but it should not be used when efficient training times are desired. Note: only set `visualize.start_server: False`
if your are certain that the server is already running, otherwise set it to `True`.
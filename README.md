# KGE: A framework for Knowledge Graph Embeddings

KGE is a framework for training, evaluating and searching for knowledge graph
embedding (KGE) models. It is based on [PyTorch](https://pytorch.org/).

# Installation

Run the following locally:

`pip install -e .`

# Quick start

The framework supports different kinds of jobs for training, evaluating and
performing hyperparameter optimization on KGE models.
The settings for a job can be specified via configuration files (YAML format).
There are many settings, all of which have default values and can be found in
[config-default.yaml](kge/config-default.yaml).

## Starting a training job

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
```
To run the job, you may specify the device and output folder:

`python kge.py start config.yaml --folder kge_test --job.device cuda:0`

All entries in the configuration file can be overwritten in the
same way. For example, to change the optimizer:

`python kge.py start config.yaml --train.optimizer Adam`

The output folder contains a complete config file with all entries
used for the job, a trace file (YAML format) with details produced
while running the job, a log file, and checkpoint files that
contain the learned model at different training epochs. The best
performing model is stored in a separate file.

## Resuming a job

All jobs can be resumed if interrupted. By default, the latest
checkpoint file is used to resume the job:

`python kge.py resume kge_test/config.yaml`

You may change the device used for running the job when resuming:

`python kge.py resume kge_test/config.yaml --job.device cuda:1`

## Running an evaluation job

Training jobs run evaluation jobs at a frequency specified in the
configuration file. To evaluate a previously trained model:

`python kge.py eval kge_test/config.yaml`

By default, the checkpoint file of the best model is used for
evaluation. To check the model's performance on test data:

`python kge.py test kge_test/config.yaml`

## Starting a search job

Search jobs are used for hyperparameter optimization. There are
several types of search jobs, e.g. grid search or bayesian optimization.
The search type and search space are specified in the configuration file.
We use [Ax](https://ax.dev/) for SOBOL (pseudo-random) and bayesian jobs.
For example, to run 10 SOBOL trials (arms) followed by 10 bayesian optimization
trials, we may specify the following configuration file:

```yaml
job.type: search
search.type: ax
dataset.name: wnrr
valid.metric: mean_reciprocal_rank_filtered
model: reciprocal_relations_model
reciprocal_relations_model.base_model.type: conve
ax_search:
  num_trials: 30
  num_sobol_trials: 10
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
To run this job, we may specify all available devices and the number of
trials to run simultaneously (evenly distributed across available devices):

`python kge.py start config.yaml --folder kge_test --search.device_pool cuda:0,cuda:1 --search.num_workers 4`

Search jobs create training jobs for each generated trial. The output folder
of a search job contains the output folder from the training job of each trial.

## Other commands
To see available commands:

`python kge.py --help`

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

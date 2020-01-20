# libKGE: A library for Knowledge Graph Embeddings

libKGE is a library for very efficient training, evaluating and tuning of [knowledge graph
embeddings](https://ieeexplore.ieee.org/document/8047276) (KGE). It is
based on [PyTorch](https://pytorch.org/) and designed to be easy to use
and easy to extend. libKGE is highly flexible for training and tuning KGE
models, as it supports many combinations of loss functions, optimizers,
training types and many more hyperparameters. 
<!--//
Hyperparameter optimization is also supported in different ways, e.g. grid search, pseudo-random search or Bayesian optimization (currently supplied by [Ax](https://ax.dev/)). 
//-->
## Feature list

 - Efficient implementations of classic and current KGE models: 
    - [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf) [(code)](kge/model/rescal.py)
    - [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) [(code)](kge/model/transe.py)
    - [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf) [(code)](kge/model/distmult.py) 
    - [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) [(code)](kge/model/complex.py)
    - [ConvE](https://arxiv.org/abs/1707.01476) [(code)](kge/model/conve.py)
 - Training:
   - Loss: Binary Cross Entropy (BCE), Kullback-Leibler Divergence (KL), Margin Ranking (MR)
   - Training types: Negative Sampling, 1vsAll, KvsAll
   - Use all optimizers and learning rate schedulers offered by PyTorch
   - Configurable early stopping
   - Configurable checkpointing
 - Hyper-parameter tuning:
   - Types: Grid, Quasi-Random (by [Ax](https://ax.dev/)), Bayesian Optimzation (by [Ax](https://ax.dev/))
   - Highly parallelizable on single machine
 - Evaluation:
   - Metrics: Mean Reciprocal Rank (MRR), HITS@k
   - Filter metrics by: relation type, relation frequency, head or tail
 - Extensive logging in machine readable format to facilitate analysis


## Results

These are some of the state-of-the-art results (w.r.t. MRR) obtained with libKGE:


<table> 
<tr><th>FB15k-237</th><th>WNRR</th></tr>
<tr>
<td>

|          | MRR       | Hits@10 |
|----------|-----------:|---------:|
| RESCAL  | 0.356      | 0.542 |
| TransE   | 0.310      | 0.493 |
| DistMult | 0.344      | 0.531 |  
| ComplEx | 0.348      | 0.536 |
| ConvE | 0.338      | 0.520 |

</td>
<td>

|          | MRR       | Hits@10 |
|----------|-----------:|---------:|
| RESCAL | 0.467      | 0.517 |
| TransE  | 0.228      | 0.519 |
| DistMult  | 0.454      | 0.535 |  
| ComplEx | 0.479      | 0.552 |
| ConvE | 0.442      | 0.505 |

</td>
</tr>
</table>


# Quick start

```
git clone https://github.com/uma-pi1/kge.git
cd kge
# install project in development mode
pip install -e .
# download and preprocess datasets
cd data
sh download_all.sh
cd ..
# train an example model on toy dataset
python kge.py start examples/toy-complex-train.yaml
```

## Configuration

libKGE supports training, evaluating and tuning KGE models. The settings for
each task can be specified with a configuration file in YAML format.
The default values and usage for available settings can be found in [config-default.yaml](kge/config-default.yaml).

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

# Known issues
- Filtering of positive samples when training with `train.type=negative_sampling` can be slow. We are currently working on a more efficient implementation.

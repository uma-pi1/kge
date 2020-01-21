# libKGE: A library for Knowledge Graph Embeddings

libKGE is a library for very efficient training, evaluation and hyperparameter optimization of [knowledge graph
embeddings](https://ieeexplore.ieee.org/document/8047276) (KGE). It is
based on [PyTorch](https://pytorch.org/) and designed to be easy to use
and easy to extend. 

<!--//
libKGE is highly flexible for training and tuning KGE
models, as it supports many combinations of loss functions, optimizers,
training types and many more hyperparameters. 
Hyperparameter optimization is also supported in different ways, e.g. grid search, pseudo-random search or Bayesian optimization (currently supplied by [Ax](https://ax.dev/)). 
//-->
## Feature list

 - **KGE models**: [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf) ([code](kge/model/rescal.py)), [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) ([code](kge/model/transe.py)), [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf) ([code](kge/model/distmult.py)), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) ([code](kge/model/complex.py)), [ConvE](https://arxiv.org/abs/1707.01476)  ([code](kge/model/conve.py))
 - **Extensive logging** in machine readable format to facilitate analysis
 - **Training**:
   - Loss: Binary Cross Entropy (BCE), Kullback-Leibler Divergence (KL), Margin Ranking (MR)
   - Training types: Negative Sampling, 1vsAll, KvsAll
   - Use all optimizers and learning rate schedulers offered by PyTorch
   - Configurable early stopping
   - Configurable checkpointing
 - **Hyper-parameter tuning**:
   - Types: Grid, Quasi-Random (by [Ax](https://ax.dev/)), Bayesian Optimzation (by [Ax](https://ax.dev/))
   - Highly parallelizable on single machine
 - **Evaluation**:
   - Entity ranking metrics: Mean Reciprocal Rank (MRR), HITS@k
   - Filter metrics by: relation type, relation frequency, head or tail


## Results

These are some of the state-of-the-art results (w.r.t. MRR) obtained with libKGE:


<table> 
<tr><th>FB15k-237</th><th>WNRR</th></tr>
<tr>
<td>

|          | MRR       | Hits@10 |
|----------|-----------:|---------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)  | 0.356      | 0.542 |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)   | 0.310      | 0.493 |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf) | 0.344      | 0.531 |  
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) | 0.348      | 0.536 |
| [ConvE](https://arxiv.org/abs/1707.01476) | 0.338      | 0.520 |

</td>
<td>

|          | MRR       | Hits@10 |
|----------|-----------:|---------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf) | 0.467      | 0.517 |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)  | 0.228      | 0.519 |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.454      | 0.535 |  
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) | 0.479      | 0.552 |
| [ConvE](https://arxiv.org/abs/1707.01476) | 0.442      | 0.505 |

</td>
</tr>
</table>


The results above where obtained by a hyper-parameter search described in our [publication](https://openreview.net/forum?id=BkxSmlBFvr).


## Quick start

```sh
# retrieve and install project in development mode
git clone https://github.com/uma-pi1/kge.git
cd kge
pip install -e .

# download and preprocess datasets
cd data
sh download_all.sh
cd ..

# train an example model on toy dataset
python kge.py start examples/toy-complex-train.yaml
```

## Configuration

libKGE supports training, evaluation and hyper-parameter tuning of KGE models. The settings for each task can be specified with a configuration file in YAML format or on the command line. The default values and usage for available settings can be found in [config-default.yaml](kge/config-default.yaml). 

#### Training a model

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
When libKGE is called with this config, it will be [expanded with all default arguments](docs/examples/expanded-config-example.yaml) from the [main default config](kge/config-default.yaml), as well the default configurations for models, for example the [model's embedder](kge/model/lookup_embedder.yaml). 

Now, to begin training, run one of the following:

```sh
# The "start" command creates an output folder local/experiments/XXXXXXXX-XXXXXX-config and begins training
# This creates an expanded version of the config file config.yaml in the output folder
python kge.py start config.yaml

# The "create" command creates an output folder but does not begin training
python kge.py create config.yaml

# Run the "resume" command, to begin or resume training on an existing output folder:
python kge.py resume local/experiments/XXXXXXXX-XXXXXX-config

# You may specify the output folder and device
python kge.py start config.yaml --folder kge_test --job.device cuda:0

# All entries in the config file can be overwritten in the command line, e.g.
python kge.py start config.yaml --train.optimizer Adam
```

#### Resume training 

All tasks can be resumed if interrupted. Run the following for the
corresponding output folder:

```sh
python kge.py resume local/experiments/XXXXXXXX-XXXXXX-config

# Change the device when resuming
python kge.py resume local/experiments/XXXXXXXX-XXXXXX-config --job.device cuda:1

```

#### Evaluate a model

To evaluate trained model, run the following:

```sh
# Evaluate a model on the validation split
python kge.py valid kge_test/config.yaml

# Evaluate a model on the test split
python kge.py test kge_test/config.yaml
```

#### Hyper-parameter optimization

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
Trials can be run in parallel across several devices:

```sh
# Run 4 trials in parallel evenly distributed across two GPUs
python kge.py resume . --search.device_pool cuda:0,cuda:1 --search.num_workers 4

# Run 3 trials in parallel, with per GPUs capacity 
python kge.py resume . --search.device_pool cuda:0,cuda:1,cuda:1 --search.num_workers 3

```

#### Export and analyse logs

Logs are stored as yaml entries for various scopes (hyper-parameter search, training, validation). libKGE provides a convience method to export the logdata to csv.


```sh
# Dump trace info for the first trial of a hyper-parameter search
python kge.py dump trace local/experiments/XXXXXXXX-XXXXXX-toy-complex-ax/00000
```

The command above yields [this CSV output](docs/examples/dump-example-model.csv)


```sh
# Dump trace info of a hyper-parameter search
python kge.py dump trace local/experiments/XXXXXXXX-XXXXXX-toy-complex-ax
```

The command above yields [this CSV output](docs/examples/dump-example-search.csv)



#### Other commands
To see all available commands:

```sh
python kge.py --help
```

## Installation

To install libKGE, clone this repository and install the requirements with pip:


```sh
git clone https://github.com/uma-pi1/kge.git
pip install -e .
```

## Supported KGE models

libKGE has implementations for the following KGE models:

- [RESCAL](kge/model/rescal.py)
- [Transe](kge/model/transe.py)
- [DistMult](kge/model/distmult.py)
- [ComplEx](kge/model/complex.py)
- [Conve](kge/model/conve.py)

The [examples](examples) folder contains some configuration files as examples of how to run these models. 

We welcome contributions to expand the list of supported models! Please see [CONTRIBUTING](CONTRIBUTING.md) for details and feel free to initially open an issue.

## Adding a new model

To add a new model to libKGE, one needs to extend the
[KgeModel](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L243)
class. A model is made up of a
[KgeEmbedder](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L170)
to associate each subject, relation and object to an embedding, and a
[KgeScorer](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L76)
to score triples.

## Known issues
- Filtering of positive samples when training with `train.type=negative_sampling` can be slow. We are currently working on a more efficient implementation.

## Other KGE frameworks and KGE implementations

Other KGE frameworks:

 - [Graphvite](https://graphvite.io/)
 - [AmpliGraph](https://github.com/Accenture/AmpliGraph)
 - [OpenKE](https://github.com/thunlp/OpenKE)
 - [PyKEEN](https://github.com/SmartDataAnalytics/PyKEEN)

KGE projects for publications that also implement a few models:
 
 - [ConvE](https://github.com/TimDettmers/ConvE)
 - [KBC](https://github.com/facebookresearch/kbc)

## How to cite

If you use our code or compare against our results please cite the following publication:

```
@inproceedings{
  ruffinelli2020you,
  title={You {\{}CAN{\}} Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings},
  author={Daniel Ruffinelli and Samuel Broscheit and Rainer Gemulla},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=BkxSmlBFvr}
}
```
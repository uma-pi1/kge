# LibKGE: A knowledge graph embedding library

LibKGE is a PyTorch-based library for efficient training, evaluation and
hyperparameter optimization of [knowledge graph
embeddings](https://ieeexplore.ieee.org/document/8047276) (KGE). It is highly
configurable, easy to use, and extensible. A major goal of LibKGE is to
facilitate reproducible research into KGE models; see our
[ICLR paper](https://github.com/uma-pi1/kge-iclr20).

## Features

 - **KGE models**: [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf) ([code](kge/model/rescal.py)), [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) ([code](kge/model/transe.py)), [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf) ([code](kge/model/distmult.py)), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) ([code](kge/model/complex.py)), [ConvE](https://arxiv.org/abs/1707.01476)  ([code](kge/model/conve.py))
 - **Training**
   - Training types: negative sampling, 1vsAll, KvsAll
   - Losses: binary cross entropy (BCE), Kullback-Leibler divergence (KL), margin ranking (MR)
   - All optimizers and learning rate schedulers of PyTorch supported
   - Early stopping
   - Checkpointing
   - Stop and resume at any time
 - **Hyperparameter tuning**
   - Grid search, manual search, quasi-random search (using [Ax](https://ax.dev/)), Bayesian
     optimization (using [Ax](https://ax.dev/))
   - Highly parallelizable (multiple CPUs/GPUs on single machine)
   - Stop and resume at any time
 - **Evaluation**
   - Entity ranking metrics: Mean Reciprocal Rank (MRR), HITS@k with/without filtering
   - Drill-down by: relation type, relation frequency, head or tail
 - **Extensive logging**
   - Logging for training, hyper-parameter tuning and evaluation in machine readable formats to facilitate analysis


## Results

Some state-of-the-art results (w.r.t. filtered MRR) obtained with LibKGE:

<table>
<tr><th>FB15k-237</th><th>WNRR</th></tr>
<tr>
<td>

|          | MRR       | Hits@10 |
|----------|-----------:|---------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)  | 0.357      | 0.541 |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)   | 0.313      | 0.497 |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf) | 0.343      | 0.531 |  
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) | 0.348      | 0.536 |
| [ConvE](https://arxiv.org/abs/1707.01476) | 0.339      | 0.521 |

</td>
<td>

|          | MRR       | Hits@10 |
|----------|-----------:|---------:|
| [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf) | 0.467      | 0.517 |
| [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)  | 0.228      | 0.520 |
| [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)  | 0.452      | 0.531 |  
| [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf) | 0.475      | 0.547 |
| [ConvE](https://arxiv.org/abs/1707.01476) | 0.442      | 0.504 |

</td>
</tr>
</table>

The results above were obtained using the hyperparameter search described in our [ICLR paper](https://openreview.net/forum?id=BkxSmlBFvr).

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

LibKGE supports training, evaluation and hyperparameter tuning of KGE models. The settings for each task can be specified with a configuration file in YAML format or on the command line. The default values and usage for available settings can be found in [config-default.yaml](kge/config-default.yaml).

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
When LibKGE is called with this config, it will be [expanded with all default arguments](docs/examples/expanded-config-example.yaml) from the [main default config](kge/config-default.yaml), as well the default configurations for models, for example the [model's embedder](kge/model/lookup_embedder.yaml).

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

#### Hyperparameter optimization

LibKGE supports various forms of hyperparameter optimization. e.g. grid
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

Logs are stored as yaml entries for various scopes (hyperparameter search, training, validation). LibKGE provides a convience method to export the logdata to csv.


```sh
# Dump trace info for the first trial of a hyperparameter search
python kge.py dump trace local/experiments/XXXXXXXX-XXXXXX-toy-complex-ax/00000
```

The command above yields [this CSV output](docs/examples/dump-example-model.csv)


```sh
# Dump trace info of a hyperparameter search
python kge.py dump trace local/experiments/XXXXXXXX-XXXXXX-toy-complex-ax
```

The command above yields [this CSV output](docs/examples/dump-example-search.csv)



#### Other commands
To see all available commands:

```sh
python kge.py --help
```

## Current Supported KGE models

LibKGE has implementations for the following KGE models:

- [RESCAL](kge/model/rescal.py)
- [Transe](kge/model/transe.py)
- [DistMult](kge/model/distmult.py)
- [ComplEx](kge/model/complex.py)
- [Conve](kge/model/conve.py)

The [examples](examples) folder contains some configuration files as examples of how to run these models.

We welcome contributions to expand the list of supported models! Please see [CONTRIBUTING](CONTRIBUTING.md) for details and feel free to initially open an issue.

## Adding a new model

To add a new model to LibKGE, one needs to extend the
[KgeModel](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L243)
class. A model is made up of a
[KgeEmbedder](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L170)
to associate each subject, relation and object to an embedding, and a
[KgeScorer](https://github.com/uma-pi1/kge/blob/1c69d8a6579d10e9d9c483994941db97e04f99b3/kge/model/kge_model.py#L76)
to score triples.

## Using a pretrained model in an application

Using a trained model trained with LibKGE is very easy. In the following example we load the best checkpoint and predict objects indexes, given a list of subject and relation indexes.  

```python
import torch
import kge.model

model : kge.model.KgeModel = kge.model.KgeModel.load_from_checkpoint('.../checkpoint_best.pt')

subject_ids = torch.Tensor([0, 2,]).long()
relation_ids = torch.Tensor([0, 1,]).long()

scores = model.score_sp(
  subject_ids, 
  relation_ids
)

object_ids = torch.argmax(scores, dim=-1)

print(object_ids)
print(model.dataset.entity_ids(subject_ids))
print(model.dataset.relation_ids(relation_ids))
print(model.dataset.entity_ids(object_ids))

# prints: 
# tensor([8399, 8855])
# ['Dominican Republic' 'Mighty Morphin Power Rangers']
# ['has form of government' 'is tv show with actor']
# ['Republic' 'Wendee Lee']
```

For other score functions (score_sp, score_po, score_so, score_spo) see [KgeModel](kge/model/kge_model.py#L455).

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

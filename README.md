# kge

# Guidelines
- Do not add any datasets or experimental code to this repository
- Code formatting using [black](https://github.com/ambv/black) and with default
  settings (line length 88)
- Code documatation following [Google Python Style Guide (Sec.
  3.8)](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings);
  see
  [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Do not use type annotation `torch.Tensor`; it is a synonym for
  torch.FloatTensor and thus does not mean "any tensor".

# Documentation

## Installation
- locally: run `pip install -e .`

## Run
- See available commands: `kge.py --help`

## Quick start
- after installation run the script `download_all.sh` in the [data/](data/) directory to download the datasets
- to train the [Complex](http://proceedings.mlr.press/v48/trouillon16.pdf) model on a toy dataset run `python kge.py start examples/toy-complex-train.yaml`
- for training it on your CPU instead of GPU run `python kge.py start examples/toy-complex-train.yaml --job.device=CPU`


## Conceptual overview
### Models
[Models](kge/model) consist of a relational scorer and embedders. Embedders can be defined separately for subjects predicates and objects. The base definitions of these classes can be found in [kge_model.py](kge/model/kge_model.py). Custom models define implementations of these classes.

### Jobs
Models and algorithms are executed by [jobs](kge/job). There are different types of jobs, for example training jobs and evaluation jobs. The respective base definitions can be found in [eval.py](kge/job/eval.py) and [train.py](kge/job/train.py). Training strategies (1toN, negative sampling...) and evaluation metrics (mean rank,...) are defined by implementations of these classes.


## Config
The framework is operated by the use of [configuration files](kge/config-default.yaml) where all the available options can be set.

## Miscellaneous
### Training job options
#### Training.type.1toN
Negative examples are created by defining all non-existing triples as negative examples if they contain a (s,p) or (p,o) pair which forms an existing triple with some o or s respectively.
This is implemented by the use of minibatches in a training epoch: (s,p) and (p,o) pairs are sampled randomly from the triples in the graph to form a minibatch. All existing triples in the graph containing these pairs are denoted as positive examples and all non-existing triples containing the pairs are denoted as negative examples.






 

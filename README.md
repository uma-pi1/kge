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
- for training it on your CPU instead of GPU run `python kge.py start examples/toy-complex-train.yaml --job.device=cpu`


## Conceptual overview
### Models
[Models](kge/model) consist of a relational scorer and embedders. Embedders can be defined separately for subjects predicates and objects. The base definitions of these classes can be found in [kge_model.py](kge/model/kge_model.py). Custom models define implementations of these classes.

### Jobs
Models and algorithms are executed by [jobs](kge/job). There are different types of jobs, for example training jobs and evaluation jobs. The respective base definitions can be found in [eval.py](kge/job/eval.py) and [train.py](kge/job/train.py). Training strategies (1toN, negative sampling...) and evaluation metrics (mean rank,...) are defined by implementations of these classes.


## Config
The framework is operated by the use of a [configuration file](kge/config-default.yaml) where one can select models, embedders, job types and all further available options. Models and embedders have specific sections in the configuration file. Below, you will find a list with the implemented models and embedders and their respective configuration defaults.

### Models

- [Rescal](kge/model/rescal.yaml)

- [DistMult](kge/model/distmult.yaml)

- [Complex model](kge/model/complex.yaml)

- [Relational Tucker3](kge/model/relational_tucker3.yaml)

- [Sparse Relational Tucker3](kge/model/sparse_relational_tucker3.yaml)

- [TransE](kge/model/transe.yaml)

- [ConvE](kge/model/conve.yaml)

- [Feed Forward Neural Net](kge/model/fnn.py)


### Embedders

 - [Lookup Embedder](kge/model/lookup_embedder.yaml)
 
 - [Projection Embedder](kge/model/projection_embedder.yaml)
 
## Miscellaneous
### Training jobs
#### Types
- train.type.1toN
    - add short description
- train.type.negative_sampling
    - add short description
### Evaluation jobs
#### Types
- eval.type.entity_ranking 
    - add short description
    
### Search jobs
#### Types
- search.type.ax
    - add short description
- search.type.manual 
    - add short description    
- search.type.grid
    - add short description
     





 

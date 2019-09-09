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
- Unspecified configuration values are indicated by
  - `''` for strings
  - `-1` for non-negative integers
  - `.nan` for floats

# Documentation
KGE is a Python library for Knowledge Graph Embeddings built on top of PyTorch.

The project is operated by the Chair of Data Analytics at the University of Mannheim.

## Installation
Locally: run `pip install -e .`

### Dependencies
kge requires:
- torch>=1.2.0
- pyyaml
- pandas
- argparse
- path.py
- ax-platform>=0.1.2
- sqlalchemy
- torchviz

## Instruction manual
Run the script `download_all.sh` in the [data](data/) directory to download and preprocess the datasets.  

The library is used by executing `kge.py`. Executing `kge.py` means to create and/or to run a job (Train, test or hyperparameter search). Configurations for the computations inside a Job, are specified in additional configuration files. The output of a job is saved in trace and log files. A log file contains the terminal output of an executed Job with a timestamp for every operation. A trace file is a set of key-value pairs, which contains statistics of events which happened in a Job. E.g., for a Train Job, this includes statistics of the epochs and evaluation rounds. Additionally, configuration files including all the configurations and checkpoints are saved. Checkpoints entail models with their parameters as well as configurations, results and other useful data. 

To execute any Job, the following debug parameters have to be set:
`<command> <additional configuration file> <configuration parameters>`

Alternatively, run in the console: 
`python3 kge.py <command> <additional configuration file> <configuration parameters>`

Commands specify which part of `kge.py` is executed. Configuration files modify different Jobs and change or supplement the default settings. Additionally, instead of specifying a configuration in the configuration file, it can be specified as a debug parameter.

### Commands	
See available commands: `kge.py --help`
- start: Start a new job (create and run it)
- create: Create a new job (but do not run it)
- resume: Resume a prior job
- eval: Evaluate the result of a prior job
- valid: Evaluate the result of a prior job using validation data
- test: Evaluate the result of a prior job using test data

### Configuration files
The framework is operated by the use of configuration files which make certain specifications for different jobs. The file [config-default.yaml](kge/config-default.yaml) contains the default configurations. Further, different models have specific configuration files and every Job can be adapted as desired with the additional configuration files. 

#### Default Configurations
The following configuration categories are specified in the default configurations (for more, see: comments in [kge](kge/config-default.yaml)):
- Job: Configurations for the Job type and the device on which the job is ran.
- dataset: Configurations for the dataset which is used.
- model: Configurations for specific models (made in the specific model configuration files).
- train: Configurations for training jobs: Training type (1toN or Negative Sampling) as well as other training parameters like no. of epochs, optimizer, learning rate scheduler, batch size, loss, etc.
- valid: Configurations for the validation of the learned models of a training job (e.g. after how many epochs to validate, validation metric, early stopping criterions).
- checkpoint: Configurations for when to create and keep a checkpoint. 
- eval: Configurations for Evaluation Jobs: Which data to evaluate on?, evaliation type, metric, etc. 
- search: Configurations for different types of hyperparamter search (Manual, Grid, or Ax Search).
- user parameters: Can be used to add additional configuration options.

#### Additional configuration files
The default configurations can be adapted for different models and jobs. Additional configurations have always be made to specify the model and its parameters. Further, whenever something has to be adapted, the default configurations are overwritten by the specified additional configurations.

### Jobs
Models and algorithms are executed by [jobs](kge/job), which can be train, train with search or evaluation. The respective base definitions can be found in [eval.py](kge/job/eval.py) and [train.py](kge/job/train.py). Training strategies (1toN, negative sampling...) and evaluation metrics (mean rank,...) are defined by implementations of these classes.

#### Type
##### Train Job (e.g. 1toN, Negative Sampling)
Trains a kge model. Usually also has evaluation jobs inside to evaluate specific epochs. The input of a train job are the preprocessed triples and the configurations. The output of a train job is saved in trace and log files. The log file (terminal output) includes the most important configurations, statistics about the used dataset as well as for the epochs and evaluation results for the epochs which were specified to be evaluated. Additionally, the used configurations and checkpoints of certain epochs and the one best epoch are saved to resume a job.
- 1toN Training:
- Negative Sampling: 

##### Train Job with Hyperparameter Search (e.g. Ax, Manual, Grid Search)
Runs train jobs with different hyperparameters, evaluates them according to a specified metric and outputs the best. A search job produces the same output as a train job for every evaluated parameter setting. Trace and Log files are created for the different settings as well as for the whole searchjob.
- Manual Search: Manually define configurations to search over
- Grid Search: Define parameters and an array of Grid-search values 
- Ax Search: Dynamic search job that picks configurations using ax

##### Evaluation Job (e.g. Entity Ranking, Triple Classification)
Evaluates the scores of a trained kge model according to the specified metric and outputs the metrics as well as job statistics in the trace.
- Entity Ranking
	- Mean Rank
	- Mean Reciprocal Rank
	- Hits@k (1, 3, 10, 50, 100, 200, 300, 400, 500, 1000)

#### Device
For running locally on CPU run `<command> <additional configuration file> --job.device=cpu` or set `job.device: cpu` in the configuration file, since the default is cuda.

### Datasets
The following datasets are currently included in the framework:
- [db100k](data/db100k) ([Ding et al. 2018](https://www.aclweb.org/anthology/P18-1011))
- [dbpedia50](data/dbpedia50) ([https://wiki.dbpedia.org/Downloads2015-04](https://wiki.dbpedia.org/Downloads2015-04))
- [fb15k](data/fb15k) ([Bordes et al. 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf))
- [fb15k-237](data/fb15k-237) ([Toutanova et al. 2015](https://www.aclweb.org/anthology/D15-1174))
- [wn18](data/wn18) ([Bordes et al. 2014](https://link.springer.com/article/10.1007/s10994-013-5363-6))
- [wnrr](data/wnrr) ([Wang et al. 2018](https://arxiv.org/pdf/1810.07180.pdf))
- [yago3-10](data/yago3-10) ([Mahdisoltani et al. 2015](https://suchanek.name/work/publications/cidr2015.pdf))
- [toy](data/toy) 

To choose a dataset, change `dataset.name` to the desired dataset. The default is a toy dataset created from the top 399 entities with the most number of triples in the training set of FB15K-237.

### Models
[Models](kge/model) consist of a relational scorer and embedders. Embedders can be defined separately for subjects predicates and objects. The base definitions of these classes can be found in [kge_model.py](kge/model/kge_model.py). Custom models define implementations of these classes.

The following models can be specified:
- [ComplEx](kge/model/complex.yaml) ([Trouillon et al. 2016](http://proceedings.mlr.press/v48/trouillon16.pdf))
- [ConvE](kge/model/conve.yaml) ([Dettmers et al. 2018](https://arxiv.org/abs/1707.01476))
- [DistMult](kge/model/distmult.yaml) ([Yang et al. 2014](https://arxiv.org/abs/1412.6575))
- [Feed Forward Neural Net](kge/model/fnn.yaml)
- [Freex](kge/model/freex.yaml)
- [RelationalTucker3](kge/model/relational_tucker3.yaml)
- [Rescal](kge/model/rescal.yaml) ([Nickel et al. 2011](https://www.researchgate.net/publication/221345089_A_Three-Way_Model_for_Collective_Learning_on_Multi-Relational_Data))
- [SparseDiagonalRescal](kge/model/sd_rescal.yaml)
- [TransE](kge/model/transe.yaml) ([Bordes et al. 2013](https://www.researchgate.net/publication/279258225_Translating_Embeddings_for_Modeling_Multi-relational_Data)

Every model has specific configurations, which can be adapted as needed.

The following embedding techniques are implemented:
 - [Lookup Embedder](kge/model/lookup_embedder.yaml) 
 - [projection_embedder](kge/model/projection_embedder.yaml)
 - [tucker3_relation_embedder](kge/model/tucker3_relation_embedder.yaml)
 - [sparse_tucker3_relation_embedder](kge/model/sparse_tucker3_relation_embedder.yaml)

All the models use the Lookup Embedder as entity embedder by default and most of the models also as relation embedder. Tucker3 Relation Embedder or Sparse Tucker3 Relation Embedder are used as relation embedder in the RelationalTucker3 model. The projection embedder can be used as relation embedder in the Sparse Diagonal Rescal Model.

import visdom
from kge.job import  TrainingJob
from kge.job import SearchJob
import numpy as np
import yaml



class VisdomHandler():
    """ Handles functionalities of Visdom visualiuations."""


    def __init__(self, job):
        self.job = job

    # TODO: some duplicate code, change when all jobs are specified
    def prepare(self):
        """ Handels the scopes and details in which Visdom tracking occurs.

         Details and scopes depend e. g. on the job type which is executed.

        """
        # pure training job
        if isinstance(self.job, TrainingJob) and self.job.parent_job == None:
            #TODO: do the env naming properly
            envname = str(self.job.config.folder).replace("_", "-")
            envname = envname.split("/")
            train_envname = envname [-1]

            self._prepare_training_job(train_envname)

        # some search job
        elif isinstance(self.job, TrainingJob) and isinstance(self.job.parent_job, SearchJob):
            # TODO: do the env naming properly
            envname = str(self.job.config.folder).replace("_", "-")
            envname = envname.split("/")

            # create a summary env (you might want to have an A at the beginning)
            summary_env_name = "A.summary plots"
            full_summary_env_name = envname[-2] + "_" + summary_env_name
            visdom.Visdom(env=full_summary_env_name)
            # underscores to generate subenvs
            train_envname = envname[-2] + "_" + envname[-1]
            self._prepare_training_job(train_envname)
            self._prepare_search_job(full_summary_env_name)


    def _prepare_search_job(self, envname):

        vis = visdom.Visdom(env=envname)
        #TODO you should do this for the best checkpoint not in the end
        def collect_hyperparam_and_valid(job,trace):
            # collect hyperparameters that are tuned in this search job
            #TODO: this is hardcoded, make this generic such that it works for all parameters that are tuned
            params = job.parent_job.config.get("grid_search.parameters.train.optimizer_args").keys()
            for param in list(params):
                x = job.config.get("train.optimizer_args.{}".format(param))
                y = trace[job.config.get("valid.metric")]
                vis.scatter(X=[[x,y]], update="append" , win=param , opts={"xlabel": param , "ylabel": job.config.get("valid.metric")  })
                #TODO adjust saving behavior
                vis.save([envname])


        self.job.post_train_hooks.append(collect_hyperparam_and_valid)


    def _prepare_training_job(self, envname):

        vis = visdom.Visdom(env=envname)

        def track_visdom_training_metrics(job, trace):
            for track_me in self.job.config.get("visdom.include.train"):
                vis.line(
                    X=[job.epoch],
                    Y=[trace[track_me]],
                    win=track_me,
                    update="append",
                    #TODO add name for legend
                    name = "add param config here",
                    opts={
                        #TODO: you obviosuly don't want to append the config every epoch again
                        "config":job.config.options,
                        "title": track_me,
                        #"width": 280,
                        #"height": 260,
                        "margintop": 35,
                        "marginbottom": 30,
                        "marginleft": 30,
                        "marginright": 30,
                        "layoutopts": {
                            "plotly": {
                                "xaxis": {
                                    "range": [0, self.job.config.get("train.max_epochs")],
                                    "autorange": False,
                                    #"showgrid": True,
                                    #"showline": True
                                },
                                "yaxis": {
                                    #"showline": True,
                                    #"showgrid": True
                                }
                            }
                        }
                    }
                )
                #TODO adjust saving behavior
                vis.save([envname])

        self.job.post_epoch_hooks.append(track_visdom_training_metrics)

        def track_visdom_eval_metrics(job, trace):
            for metric in trace.keys():
                for track_me in self.job.config.get("visdom.include.eval"):
                    if track_me in metric:
                        vis.line(
                            X=[job.epoch],
                            Y=[trace[metric]],
                            win=metric,
                            update="append",
                            # TODO add name for legend
                            name="add param config here",
                            opts={
                                #TODO: you obviously don't want to append the config everytime again
                                "config": job.config.options,
                                "title": metric,
                                #"width": 280,
                                #"height": 260,
                                "margintop": 35,
                                "marginbottom": 30,
                                "marginleft": 30,
                                "marginright": 30,
                                "layoutopts": {
                                    "plotly": {
                                        "xaxis": {
                                            "range": [0, self.job.config.get("train.max_epochs")],
                                            "autorange": False,
                                            #"showline":True,
                                            #"showgrid":True
                                        },
                                        "yaxis": {
                                            #"showline": True,
                                            #"showgrid": True
                                        }
                                    }
                                }
                            }
                        )
                        # TODO adjust saving behavior
                        vis.save([envname])
        self.job.valid_job.post_valid_hooks.append(track_visdom_eval_metrics)

        def add_config(job, trace_entry):
            vis.text(yaml.dump(job.config.options).replace("\n", "<br>"))
            vis.save([envname])
        self.job.post_train_hooks.append(add_config)
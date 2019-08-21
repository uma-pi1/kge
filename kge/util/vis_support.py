import visdom
from kge.job import TrainingJob
from kge.job import SearchJob
import numpy as np
import yaml
import json



class VisdomHandler():
    """ Handles functionalities of interactive visualizations of job executions."""

    def __init__(self, job):
        self.job = job
        # train config is appended to every window once
        self.config_appended_eval = False
        self.config_appended_train = False

        self.full_summary_env_name = None
        self.train_envname = None
        # best valid.metric of the training job using this handler
        self.best_valid_metric = None

        #TODO change this
        self.sub_name = None

    # TODO: some duplicate code, change when all jobs are specified
    def prepare(self):
        """ Handels the scopes and details in which Visdom tracking occurs.

         Depending on job type and details _prepare_x() methods  are called to register hooks.

        """
        # pure training job
        if isinstance(self.job, TrainingJob) and self.job.parent_job == None:
            #TODO: do the env naming properly
            envname = str(self.job.config.folder).replace("_", "-")
            envname = envname.split("/")
            train_envname = envname[-1]

            self._prepare_training_job(train_envname)

        # some search job
        elif isinstance(self.job, TrainingJob) and isinstance(self.job.parent_job, SearchJob):
            # TODO: do the env naming properly
            envname = str(self.job.config.folder).replace("_", "-")
            envname = envname.split("/")

            # create a summary env (you might want to have an A at the beginning)
            summary_env_name = "A.summary plots"
            self.full_summary_env_name = envname[-2] + "_" + summary_env_name
            visdom.Visdom(env=self.full_summary_env_name)
            # underscores to generate subenvs
            self.train_envname = envname[-2] + "_" + envname[-1]
            self.sub_name = envname[-1]
            self._prepare_training_job(self.train_envname)
            self._prepare_search_scopes(self.full_summary_env_name)

    def _prepare_search_scopes(self, envname):
        """ Registers Visdom hooks for training jobs that are run by a search job.

        Metrics for the search job with scope=train and scope=search are collected and written to the summary
        environment of the search job. See manual_search.py for a description of the scopes.

        """
        vis = visdom.Visdom(env=envname)

        #TODO you should do this for the best checkpoint not in the end
        def collect_hyperparam_and_valid(job,trace):
            # collects hyperparameters that are tuned in this search job
            #TODO: this is hardcoded, make this generic such that it works for all parameters that are tuned
            params = job.parent_job.config.get("grid_search.parameters.train.optimizer_args").keys()
            for param in list(params):
                x = job.config.get("train.optimizer_args.{}".format(param))
                y = trace[job.config.get("valid.metric")]
                vis.scatter(
                    X=[[x,y]],
                    update="append",
                    win=param,
                    opts={
                        "xlabel": param,
                        "ylabel": job.config.get("valid.metric")
                    }
                )
                #TODO adjust saving behavior
                vis.save([envname])
        self.job.post_train_hooks.append(collect_hyperparam_and_valid)


        def track_best_valid(job, trace_entry):

            valid_metric_name = self.job.config.get("valid.metric")
            valid_metric = trace_entry[valid_metric_name]

            title = "bar"
            #TODO Note checking for the best valid is done at two positions in the script
            # TrainingJob.run before an epoch "best_index"
            # and search.py in run_train_job (
            # both are not reachable at the moment

            # first check if you update the training jobs's bar
            if self.best_valid_metric == None:
                self.best_valid_metric = valid_metric

                # you write to it for the first time
                if vis.win_exists(title):
                    data = vis.get_window_data(title)
                    js = json.loads(data)
                    bars = js["content"]["data"]
                    bars.append({
                        "type": "bar",
                         "name": self.train_envname,
                         "x": [self.sub_name],
                         "y": [valid_metric]}
                    )
                # create the window
                else:
                    bars = [{
                            "type": "bar",
                            "name": self.train_envname,
                            "x": [self.sub_name],
                            "y": [valid_metric]}
                    ]
                fig = {
                    "data": bars,
                    "layout": {"title": {"text": title}},
                    "win": title
                }
                vis._send(fig)

            # window exists because at least you created it
            else:
                if valid_metric > self.best_valid_metric:
                    self.best_valid_metric = valid_metric
                    data = vis.get_window_data(title)
                    js = json.loads(data)
                    bars = js["content"]["data"]
                    my_bar_index = bars.index(list(filter(lambda adic: adic["name"] == self.train_envname, bars))[0])
                    # update your bar in the bar plot
                    bars[my_bar_index] = {
                            "type": "bar",
                            "name": self.train_envname,
                            "x": [self.sub_name],
                            "y": [valid_metric]}
                    fig = {
                        "data": bars,
                        "layout": {"title": {"text": title}},
                        "win": title
                    }
                    vis._send(fig)


            #TODO indent this, you only have to check for overall improvement if the training job

            # improved itself in the first place
            # now check if there is improvement on overall search scope
            title = "progress_best_{}".format(valid_metric_name)
            # TODO: the ops somehow lead to a weird appearance
            # opts = self._get_opts(title)
            # opts["layoutopts"]["plotly"]["xaxis"].pop("range")
            # opts["layoutopts"]["plotly"]["xaxis"]["autorange"] = True

            if (vis.win_exists(title)):
                data = vis.get_window_data(title)
                js = json.loads(data)
                last_val = js["content"]["data"][0]["y"][-1]
                step_num = js["content"]["data"][0]["x"][-1]
                if valid_metric > last_val:
                    self._vis_line(
                        vis=vis,
                        X=[step_num + 1],
                        Y=[valid_metric],
                        win=title,
                        opts={"title": title}
                    )
            else:
                self._vis_line(
                    vis=vis,
                    X=[self.job.epoch],
                    Y=[valid_metric],
                    win=title,
                    opts={"title": title}
                )
        self.job.post_valid_hooks.append(track_best_valid)

    def _prepare_training_job(self, envname):
        """ Registers Visdom hooks for a training job.

        The function can be used for any training job in any setting. It does not care if the training job has
        a parent search job.

        """
        vis = visdom.Visdom(env=envname)

        def track_visdom_training_metrics(job, trace):
            for track_me in self.job.config.get("visdom.include.train"):
                # create or update plot for train metrics
                self._vis_line(
                    vis=vis,
                    X=[job.epoch],
                    Y=[trace[track_me]],
                    win=track_me,
                    opts=self._get_opts(track_me),

                )
                if not self.config_appended_train:
                    # appends the config as meta data to the window this updates opts and does not override it
                    # meta data will appear as a key "config":{} in the "layout" dic e. g. in vis.get_window_data(winid)
                    # it is fine to "abuse" opts in that way, see:
                    # https://github.com/facebookresearch/visdom/issues/661
                    vis.line(
                        X=[None],
                        Y=[None],
                        win=track_me,
                        update="append",
                        opts={"config": self.job.config.options}
                    )
            self.config_appended_train = True
            #TODO adjust saving behavior
            vis.save([envname])
        self.job.post_epoch_hooks.append(track_visdom_training_metrics)

        def track_visdom_eval_metrics(job, trace):
            for metric in trace.keys():
                for track_me in self.job.config.get("visdom.include.eval"):
                    if track_me in metric:
                        # create or update plot for valid metrics
                        self._vis_line(
                            vis=vis,
                            X=[job.epoch],
                            Y=[trace[metric]],
                            win=metric,
                            opts = self._get_opts(title=metric)
                        )
                        # appends the config as meta data to the window this updates opts and does not override it
                        # meta data will appear as a key "config":{} in the "layout" dic e. g. in vis.get_window_data(winid)
                        # it is fine to "abuse" opts in that way, see:
                        # https://github.com/facebookresearch/visdom/issues/661
                        if not self.config_appended_eval:
                            vis.line(
                                X=[None],
                                Y=[None],
                                win=metric,
                                update="append",
                                opts={"config": self.job.config.options}
                            )
            self.config_appended_eval = True
            # TODO adjust saving behavior
            vis.save([envname])
        self.job.valid_job.post_valid_hooks.append(track_visdom_eval_metrics)

        def add_config(job, trace_entry):
            # adds the config as a window to the environment
            vis.text(yaml.dump(job.config.options).replace("\n", "<br>"))
            vis.save([envname])
        self.job.post_train_hooks.append(add_config)


    def _vis_line(self, vis, X, Y, win, opts):
        """ Wraps vis.line(). """
        vis.line(
            X=X,
            Y=Y,
            win=win,
            update="append",
            # TODO add name for legend
            name="add param config here",
            opts=opts)

    def _get_opts(self, title):
        opts = {
                "title": title,
                # "width": 280,
                # "height": 260,
                "margintop": 35,
                "marginbottom": 30,
                "marginleft": 30,
                "marginright": 30,
                "layoutopts": {
                    "plotly": {
                        "xaxis": {
                            "range": [0, self.job.config.get("train.max_epochs")],
                            "autorange": False,
                            # "showline":True,
                            # "showgrid":True
                        },
                        "yaxis": {
                            # "showline": True,
                            # "showgrid": True
                        }
                    }
                }
            }
        return opts




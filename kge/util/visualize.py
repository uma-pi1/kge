from kge.job.trace import Trace
from kge.job import Job, TrainingJob, SearchJob, Trace
from kge.util.misc import kge_base_dir
from kge.model.kge_model import KgeModel
from kge.model import ReciprocalRelationsModel
from kge.config import Config
from kge import Dataset
import os
import yaml
import re
import json
import copy
from collections import defaultdict
import numpy as np
import random


class VisualizationHandler:
    """ Base class for broadcasting and post-processing that contains base functionalities for interacting with data
    produced by the kge framework.

    Subclasses implement create(), _visualize_train_item() _visualize_eval_item(), post_process_trace() (and optionally
    _process_search_trace_entry())

    writer: A writer object that writes data to a plotting framework
    path: path of the job folder which is currently visualized; can change during a session
    session_type: "post" or "broadcast"
    include_train: metrics from a training job to be included in the visualizations
    include_eval: metrics from a eval job to be included in the visualizations
    exclude_train: metrics to exclude; has priority over include train
    exclude_eval: metrics to exclude; has priority over include eval
    session_config: the options of the current visualization session; does not change in a session. In a broadcast
    session the session_config = job config of the job that is run by the kge framework
    session_data: can be used to cache arbitrary data

     """

    def __init__(self,
                 writer,
                 path=None,
                 session_type=None,
                 include_train=None,
                 include_eval=None,
                 exclude_train=None,
                 exclude_eval=None,
                 session_config=None,
                 session_data={}
    ):
        self.writer = writer
        self.session_type = session_type
        self.include_train = include_train
        self.include_eval = include_eval
        self.exclude_train = exclude_train
        self.exclude_eval = exclude_eval
        # session data can be used to cache any kind of data that is needed during a visualization session
        # during broadcasting this can be best valid.metric during post processing this can be metadata for envs etc.
        self.session_data = session_data
        self.path = path
        self.session_config = session_config

    @classmethod
    def create_handler(cls, **kwargs):
        if kwargs["module"] == "visdom":
            return VisdomHandler.create(**kwargs)
        elif kwargs["module"] == "tensorboard":
            return TensorboardHandler.create(**kwargs)

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    def _process_search_trace_entry(self, trace_entry):
        raise NotImplementedError

    def post_process_trace(self, tracefile, tracetype, jobtype, subjobs=None, **kwargs):
        raise NotImplementedError

    def _visualize_config(self, **kwargs):
        raise NotImplementedError

    def process_trace(self, tracefile, tracetype, jobtype):
        """ Loads a trace file and processes it.

        :param tracefile:
        :param tracetype: "search", "train", "eval" the type of the trace, this is independent of jobtype because the
        overall jobtype can be e. g. search which also has train type trace files.
        :param jobtype "search", "train", "eval"
        """
        # group traces into dics with {key: list[values]}
        # grouped_entries contains for every distinct trace entry type (train,eval, metadata..)
        # a dic where the keys are the original keys and the values are lists of the original values
        grouped_entries = []
        with open(tracefile, "r") as file:
            raw = file.readline()
            while (raw):
                trace_entry = yaml.safe_load(raw)
                group_entry = defaultdict(list)
                # add the very first entry
                if len(grouped_entries) == 0:
                    [group_entry[k].append(v) for k, v in trace_entry.items()]
                    grouped_entries.append(group_entry)
                else:
                    matched = False
                    for entry in grouped_entries:
                        if entry.keys() == trace_entry.keys():
                            e_scope = entry.get("scope")
                            e_job = entry.get("job")
                            t_scope = trace_entry.get("scope")
                            t_job = trace_entry.get("job")
                            # check if scope and job is the same
                            if e_scope and e_job and t_scope and t_job:
                                if not (e_scope[0] == t_scope and e_job[0] == t_job):
                                    break
                            # append the data as the entry type exists already
                            [entry[k].append(v) for k, v in trace_entry.items()]
                            matched = True
                            break
                    # create new entry type as entry does not exist
                    if matched == False:
                        [group_entry[k].append(v) for k, v in trace_entry.items()]
                        grouped_entries.append(group_entry)
                raw = file.readline()
        [self._process_trace_entry(entry, tracetype, jobtype) for entry in grouped_entries]

    def _process_trace_entry(self, trace_entry, tracetype, jobtype):
        """ Process some trace entry which can be grouped or not.

       Note that there are different settings which can occur: E. g. a searchjob can have training traces which also have
       eval entries. This has to be tracked because then depending on "broadcast" or "post" behavior differs in the
       various settings.

        """
        entry_keys = list(trace_entry.keys())
        epoch = trace_entry.get("epoch")
        if epoch:
            entry_type = None
            scope = None
            # entries can be grouped or single entries
            if type(trace_entry["job"]) == list:
                scope = trace_entry["scope"][0]
                entry_type = trace_entry["job"][0]
            else:
                entry_type = trace_entry["job"]
                scope = trace_entry["scope"]
            # catch and ignore 'example' or 'batch' scopes for now
            if scope == "example" or scope == "batch":
                return
            include_patterns = []
            if entry_type == "train" and tracetype == "train":
                include_patterns = self.include_train
                exclude_patterns = self.exclude_train
                visualize = self._visualize_train_item
            elif entry_type == "eval" and tracetype == "train":
                include_patterns = self.include_eval
                exclude_patterns = self.exclude_train
                visualize = self._visualize_eval_item
            # just send search trace entries trough
            elif tracetype == "search":
                self._process_search_trace_entry(trace_entry)
                return
            # filter keys and send keys to submethods to deal with the visualizations
            matched_keys = self.filter_entry_keys(include_patterns, exclude_patterns, entry_keys)
            list(map(
                lambda matched: visualize(matched, trace_entry[matched], epoch, tracetype, jobtype),
                matched_keys
            ))

    def filter_entry_keys(self, include_patterns, exclude_patterns, keys):
        """ Returns matched keys.

        Takes a list of include_patterns, exlude_patterns and keys. Returns a list of keys
        that have some regex match with any of the include_patterns but no match with any of the exclude patterns.

        """
        return [
            matched_key
            for exclude_pattern in exclude_patterns
            for include_pattern in include_patterns
            for matched_key in list(filter(lambda match_key: re.search(include_pattern, match_key), keys))
            if matched_key not in list(filter(lambda match_key: re.search(exclude_pattern, match_key), keys))

        ]

    @classmethod
    def register_broadcast(cls, **kwargs):
        """ Bundles the different information that are needed to perform broadcasting and registers hooks.

        The user parameter inputs are collected and depending on the jobtype, broadcasting functionality is
        registered as hooks.

        """
        # called once on job creation
        def init_hooks(job):

            if isinstance(job, TrainingJob):
                jobpath = str(job.config.folder)
                tracetype = None
                jobtype = None
                # pure training job
                if job.parent_job == None:
                    tracetype = "train"
                    jobtype = "train"
                    session_data = {}
               # some search job
                if isinstance(job.parent_job, SearchJob):
                    tracetype = "train"
                    jobtype = "search"
                    session_data = {"valid_metric_name":job.config.get("valid.metric")}

                handler = VisualizationHandler.create_handler(
                    module=kwargs["module"],
                    session_type=kwargs["session_type"],
                    include_train=kwargs["include_train"],
                    include_eval=kwargs["include_eval"],
                    exclude_train=kwargs["exclude_train"],
                    exclude_eval=kwargs["exclude_eval"],
                    path=jobpath,
                    session_data=session_data,
                    session_config=job.config.options
                )
                handler._visualize_config(
                    env=handler.get_env_from_path(tracetype,jobtype),
                    job_config=handler.session_config #in a broadcast session it holds session_config = job_config
                )
                def visualize_data(job, trace_entry):
                    handler._process_trace_entry(trace_entry, tracetype=tracetype, jobtype=jobtype)
                job.post_epoch_hooks.append(visualize_data)
                job.valid_job.post_valid_hooks.append(visualize_data)

        Job.job_created_hooks.append(init_hooks)

    def _get_job_config(self, path=None):
        """ Returns the config of the current job which is visualized.

        This is different from the session-config, e. g. the visualization options of
        the current visualization session. In "broadcasting", both are the same.

        """
        job_config = None
        if path ==None:
            job_config = self.path + "/" + "config.yaml"
        else:
            job_config = path + "/" + "config.yaml"

        if os.path.isfile(job_config):
            with open(job_config, "r") as file:
                job_config = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            raise Exception("Could not find config.yaml in path.")
        return job_config

    @classmethod
    def post_process_jobs(cls, **kwargs):
        """ Scans all the executed jobs in local/experiments and allows submodules to deal with them as they please. """

        path = kge_base_dir() + "/local/experiments/"

        handler = VisualizationHandler.create_handler(
            module=kwargs["module"],
            session_type="post",
            include_train=kwargs["include_train"],
            include_eval=kwargs["include_eval"],
            exclude_train=kwargs["exclude_train"],
            exclude_eval=kwargs["include_eval"],
        )
        for parent_job in os.listdir(path):
            if parent_job not in kwargs["folders"] and not len(kwargs["folders"]) == 0:
                continue
            parent_trace = path + parent_job + "/trace.yaml"
            first_entry = None
            job_config = path + parent_job + "/config.yaml"
            if os.path.isfile(parent_trace) and os.path.isfile(job_config):
                with open(job_config, "r") as conf:
                    job_config = yaml.load(conf, Loader=yaml.SafeLoader)
                # pure training job
                if job_config["job"]["type"] == "train":
                    handler.path = path + parent_job
                    handler.post_process_trace(parent_trace, "train", "train", **kwargs)
                elif job_config["job"]["type"] == "search":
                    # collect the training sub job folders by searching through the parent directory
                    subjobs = [
                                path + parent_job + "/" + child_folder for child_folder in list(
                                    filter(
                                        lambda file:
                                            os.path.isdir(path + parent_job + "/" + file)
                                            and file != "config"
                                            and "trace.yaml" in os.listdir(path + parent_job + "/" + file),
                                        os.listdir(path + parent_job)
                                    )
                                )
                    ]
                    handler.path = path + parent_job
                    handler.post_process_trace(parent_trace, "search", "search", subjobs, **kwargs)
        input("Post processing finished.")


class VisdomHandler(VisualizationHandler):
    def __init__(
            self,
            writer,
            session_type=None,
            include_train=None,
            include_eval=None,
            exclude_train =None,
            exclude_eval=None,
            path=None,
            session_config=None,
            session_data={}):

        super().__init__(
            writer,
            path,
            session_type,
            include_train,
            include_eval,
            exclude_train,
            exclude_eval,
            session_config,
            session_data)

    @classmethod
    def create(cls, **kwargs):
        vis = visdom.Visdom()
        return VisdomHandler(
            vis,
            session_type=kwargs.get("session_type"),
            include_train=kwargs.get("include_train"),
            include_eval=kwargs.get("include_eval"),
            exclude_train=kwargs.get("exclude_train"),
            exclude_eval=kwargs.get("exclude_eval"),
            path=kwargs.get("path"),
            session_data=kwargs.get("session_data"),
        )

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        env = self.get_env_from_path(tracetype, jobtype)
        key = key + " (train)"
        if jobtype == "train":
            self._visualize_item(key, value, epoch, env=env)
        elif jobtype == "search":
            self._visualize_item(key, value, epoch, env=env)

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        env = self.get_env_from_path(tracetype, jobtype)
        key = key + " (eval)"
        if jobtype == "train":
            self._visualize_item(key, value, epoch, env=env)
        elif jobtype == "search" and tracetype == "train":
            self._visualize_item(key, value, epoch, env=env)
            if key == self.session_data["valid_metric_name"] + " (eval)":
                if self.session_type == "broadcast":
                    # progress plot
                    self._track_progress(key, value, env)
                # plot of all valid.metrics in summary env
                self._visualize_item(
                    key,
                    value,
                    epoch,
                    env=self.extract_summary_envname(env),
                    name=env.split("_")[-1],
                    title="all_" + key,
                    legend=[env.split("_")[-1]],
                    win="all_" + key
                )

    def _process_search_trace_entry(self, trace_entry):
        # this function only works for grouped trace entries, which is fine as it is only used in post processing
        if "train" in trace_entry["scope"]:
            # bar plot for valid.metric
            valid_metric_name = self.session_data["valid_metric_name"]
            x = trace_entry[valid_metric_name]
            names = trace_entry["folder"]
            env = self.extract_summary_envname(envname=self.get_env_from_path("search","search"),jobtype="search")
            self.writer.bar(
                X=x,
                env=env,
                opts={"legend":names, "title": valid_metric_name + "_best"}
            )
            params = list(filter(lambda key: "." in key, list(trace_entry.keys())))
            # bar plots for tuned paramter values of the trials (job vs parameter value)
            #  +  scatters valid.metric vs parameter values of trials (scope: train)
            for param in params:
                x = trace_entry[param]
                if type(x[0]) == list or type(x[0]) == str:
                    return
                self.writer.scatter(
                    X=np.array([x, trace_entry[valid_metric_name]]).T,
                    opts={"title": "Best valid vs " + param.split(".")[-1], "xlabel": param, "ylabel": valid_metric_name},
                    env=env
                )
                self.writer.bar(
                    X=x,
                    env=env,
                    opts={"legend":names, "title": param}
                )


    def _visualize_item(self, key, value, x, env, name=None , win=None, update="append", title=None, **kwargs):
        if win == None:
            win = self.extract_window_name(env, key)
        if name == None:
            name = env.split("_")[-1]
        if title == None:
            title = key
        x_ = None
        val = None

        if type(x) == list:
            x_ = x
            val = value
        else:
            x_ = [x]
            val = [value]

        # skip datatypes that cannot be handled at the moment to prevent errors
        if type(val[0]) == list or type(val[0]) == str:
            return

        if not self.writer.win_exists(win, env):
            self.writer.env = env
            self.writer.line(
                X=x_,
                Y=val,
                win=win,
                opts=self._get_opts(title=title, **kwargs),
                name=name,
            )
        else:
            self.writer.env = env
            self.writer.line(
                X=x_,
                Y=val,
                win=win,
                opts=self._get_opts(title=title),
                name=name,
                update=update
            )

    def _visualize_config(self, env, job_config=None):
        if job_config == None:
            job_config = self._get_job_config()
        self.writer.text(yaml.dump(job_config).replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"), env=env)

    def _track_progress(self, key, value, env):
        """ Updates the overall progress plot of a key over multiple train jobs in a search job."""

        # Value is updated whenever a higher value than the current value is found.
        # This is used for valid.metric but can potentially also be used for other keys.
        env = self.extract_summary_envname(env)
        title = "progress_best_{}".format(key)
        win = self.extract_window_name(env, key)
        check = False
        best = self.session_data.get("best_{}".format(key))
        if best and value > best:
           self.session_data ["best_{}".format(key)] = value
           check = True
        elif not best:
           self.session_data["best_{}".format(key)] = value
           check = True
        if not self.writer.win_exists(win, env):
            self._visualize_item(title, value, 1, env=env,win=win)
        elif check == True and (self.writer.win_exists(win,env)):
            data = self.writer.get_window_data(win, env)
            js = json.loads(data)
            best_val = js["content"]["data"][0]["y"][-1]
            step_num = js["content"]["data"][0]["x"][-1]
            if value > best_val:
                self._visualize_item(title, value, step_num+1, env=env, win=win)

    def post_process_trace(self, tracefile, tracetype, jobtype, subjobs=None, **kwargs):
        """ Creates an empty environment with a properties button 'sync' which loads the data in the environment. """

        properties = [
            {'type': 'button', 'name': 'Click to sync env', 'value': 'Synchronize'}
        ]
        # this returns just the string id of the window
        #TODO maybe add a callback to delete the property window after sync has been pressed (no overlapoing windows?)
        path = copy.deepcopy(self.path)
        properties_window = None
        if jobtype == "train":
            # create window with sync
            properties_window = self.writer.properties(properties, env=self.get_env_from_path("train", "train"))
            self._visualize_config(env=self.get_env_from_path("train", "train"))
            def properties_callback(event):
                if event['event_type'] == 'PropertyUpdate':
                    env = event["eid"]
                    # reset the path because when this function is called it might have changed
                    self.path = path
                    self.process_trace(tracefile, tracetype, jobtype)
            self.writer.register_event_handler(properties_callback, properties_window)
        elif jobtype =="search":
            env = self.extract_summary_envname(self.get_env_from_path("search", "search"), "search")
            properties_window = self.writer.properties(properties, env=env)
            self._visualize_config(env)
            def properties_callback(event):
                if event['event_type'] == 'PropertyUpdate':
                    self.path = path
                    self.session_data = {"valid_metric_name": self._get_job_config().get("valid")["metric"]}
                    self.process_trace(tracefile, "search", "search")
                    # process training jobs of the search job
                    for subjob in sorted(subjobs):
                        self.path = subjob
                        self._visualize_config(self.get_env_from_path("train", "search"))
                        self.session_data = {"valid_metric_name": self._get_job_config().get("valid")["metric"]}
                        self.process_trace(subjob + "/" + "trace.yaml", "train", "search")
            self.writer.register_event_handler(properties_callback, properties_window)

    def extract_summary_envname(self, envname, jobtype="train"):
        """ Extracts the summary environment name.
        jobtype is train if it's a training job who belong to  a search job.
        """
        if jobtype == "train":
            env = envname.split("_")[-2]
            env = env + "_" + "SummaryEnvironment"
        elif jobtype == "search":
            env = envname.split("_")[-1]
            env = env + "_" + "SummaryEnvironment"
        return env

    def get_env_from_path(self, tracetype, jobtype):
        path = self.path.replace("_", "-")
        parts = path.split("/")
        if tracetype == "train" and jobtype == "search":
            return parts[-2] + "_" + parts[-1]
        elif tracetype == "train" and jobtype == "train":
            return parts[-1]
        elif jobtype == "search":
            return parts[-1]

    def extract_window_name(self, env, key):
        # initially this was "key + "_" + env" but then windows overlapped somehow in the 'compare envs' functionality
        # seems fine to have same window names over multiple environments
        return key

    def _get_opts(self, title, **kwargs):
        opts = {
            #"ylabel"
            "title": title,
            # "width": 280,
            # "height": 260,
            # "margintop": 35,
            "marginbottom": 30,
            "marginleft": 30,
            "marginright": 30,
            "layoutopts": {
                "plotly": {
                    "xaxis": {
                    #"range": [0, 200],
                    "autorange": True,
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
        for key,value in kwargs.items():
            opts[key] = value
        return opts

    @classmethod
    def run_server(cls, session_config, **kwargs):
        import subprocess
        import sys, time
        import signal
        import atexit
        import socket

        envpath = kge_base_dir() + "/local/visualize/visdomenvs"
        if not os.path.isdir(envpath):
            os.mkdir(envpath)

        DEFAULT_PORT = 8097 #must not be changend atm
        DEFAULT_HOSTNAME = "localhost"

        # restart the server if it is running to not mix up data from different sessions
        subprocess.Popen(
            ["pkill visdom"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        stdout = None
        stderr = None
        if session_config.get("visualize.surpress_server_output"):
            stdout, stderr = subprocess.DEVNULL, subprocess.STDOUT
        # run server
        process = subprocess.Popen(
            ["visdom" + " -env_path=" + envpath + " --hostname=" + DEFAULT_HOSTNAME + " -port=" + str(DEFAULT_PORT)],
            shell=True,
            stdout=stdout,
            stderr=stderr
        )
        time.sleep(1)
        print("Visdom running on http://{}:{}".format(DEFAULT_HOSTNAME, DEFAULT_PORT))
        # kill server when everyone's done
        server_pid = process.pid
        def kill_server():
            if server_pid:
                os.kill(server_pid, signal.SIGTERM)
        atexit.register(kill_server)


class TensorboardHandler(VisualizationHandler):
    def __init__(
            self,
            writer,
            session_type=None,
            include_train=None,
            include_eval=None,
            exclude_train=None,
            exclude_eval=None,
            path=None,
            session_config=None,
            session_data={},
    ):

        super().__init__(
            writer,
            path,
            session_type,
            include_train,
            include_eval,
            exclude_train,
            exclude_eval,
            session_config,
            session_data)
        self.writer_path = kge_base_dir() + "/local/visualize/tensorboard/"

    @classmethod
    def create(cls, **kwargs):
        writer = tensorboard.SummaryWriter(kge_base_dir() + "/local/visualize/tensorboard_remove/")
        return TensorboardHandler(
            writer,
            session_type=kwargs.get("session_type"),
            include_train=kwargs.get("include_train"),
            include_eval=kwargs.get("include_eval"),
            exclude_train=kwargs.get("exclude_train"),
            exclude_eval=kwargs.get("exclude_eval"),
            path=kwargs.get("path"),
            session_data=kwargs.get("session_data"),
        )

    def _add_embeddings(self, path):
        checkpoint = self._get_checkpoint_path(path)
        if checkpoint:
            model = KgeModel.load_from_checkpoint(checkpoint)
            meta_ent = model.dataset.entities
            meta_rel = model.dataset.relations
            if isinstance(model, ReciprocalRelationsModel):
                meta_ent = None
                meta_rel = None

            self.writer.add_embedding(
                mat=model.get_s_embedder().embed_all(),
                metadata=meta_ent,
                tag="Entity embeddings"
            )
            self.writer.add_embedding(
                mat=model.get_p_embedder().embed_all(),
                metadata=meta_rel,
                tag="Relation embeddings"
            )
            del model

    def _get_checkpoint_path(self, path):
        if "checkpoint_best.pt" in os.listdir(path):
            return path + "/" + "checkpoint_best.pt"
        else:
            checkpoints = sorted(list(filter(lambda file: "checkpoint" in file, os.listdir(path))))
            if len(checkpoints) > 0:
                return path + "/" + checkpoints[-1]
            else:
                print("Skipping {} for embeddings as no checkpoint could be found.".format(path))

    def post_process_trace(self, tracefile, tracetype, jobtype, subjobs=None, **kwargs):
        self.writer.close()
        if jobtype == "train":
            event_path = self.writer_path + self.path.split("/")[-1]
            self.writer.log_dir = event_path
            self._visualize_config(path=self.path)
            self.process_trace(tracefile, tracetype, jobtype)
            if kwargs["embeddings"]:
                self._add_embeddings(self.path)
            self.writer.close()
        elif jobtype == "search":
            for subjob_path in subjobs:
                event_path = self.writer_path + subjob_path.split("/")[-2] + "/" + subjob_path.split("/")[-1]
                self.writer.log_dir = event_path
                self._visualize_config(path=subjob_path)
                self.process_trace(subjob_path + "/" + "trace.yaml", "train", "search")
                if kwargs["embeddings"]:
                    self._add_embeddings(subjob_path)
                self.writer.close()

    def _visualize_config(self, **kwargs):
        config = self._get_job_config(kwargs.get("path"))
        self.writer.add_text(
            "config",
            yaml.dump(config).replace("  ", "&nbsp;&nbsp;&nbsp;").replace("\n", "  \n"),
        )



    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        key = key + " (train)"
        # skip datatypes that cannot be handled
        if type(value[0]) == list or type(value[0]) == str:
            return
        if len(value)>1:
            idx = 0
            # TODO apparantly tensorboard cannot handle lists
            for val in value:
                self.writer.add_scalar(key, val, epoch[idx])
                idx += 1

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        key = key + " (eval)"
        # skip datatypes that cannot be handled
        if type(value[0]) == list or type(value[0]) == str:
            return
        if len(value)>1:
            idx = 0
            # TODO apparantly tensorboard cannot handle lists
            for val in value:
                self.writer.add_scalar(key, val, epoch[idx])
                idx += 1

    @classmethod
    def run_server(cls, session_config, **kwargs):
        import subprocess
        import time
        import atexit
        import signal
        import shutil

        folders = session_config.get("visualize.post.folders")
        # clean up log_dir from previous visualization
        logdir = kge_base_dir() + "/local/visualize/tensorboard"
        if os.path.isdir(logdir):
            for folder in os.listdir(logdir):
                shutil.rmtree(logdir + "/" + folder)

        DEFAULT_PORT = 6006  # set to different port than the visdom port
        DEFAULT_HOSTNAME = "localhost"

        # restart the server if it is running to not mix up data from different sessions
        subprocess.Popen(
            ["pkill tensorboard"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        stdout = None
        stderr = None
        if session_config.get("visualize.surpress_server_output"):
            stdout, stderr = subprocess.DEVNULL, subprocess.STDOUT
        # run server
        process = subprocess.Popen(
            ["tensorboard --logdir={} --port={} --host={}".format(logdir, DEFAULT_PORT, DEFAULT_HOSTNAME)],
            shell=True,
            stdout=stdout,
            stderr=stderr
        )
        time.sleep(1)
        print("Tensorboard running on http://{}:{}".format(DEFAULT_HOSTNAME, DEFAULT_PORT))
        # kill server when everyone's done
        server_pid = process.pid
        def kill_server():
            if server_pid:
                os.kill(server_pid, signal.SIGTERM)
        atexit.register(kill_server)

def initialize_visualization(session_config, command):

    include_train = session_config.get("visualize.include_train")
    exclude_train = session_config.get("visualize.exclude_train")
    if len(include_train) == 0:
        include_train = [""]
    if len(exclude_train) == 0:
        exclude_train = [str(random.getrandbits(64))]
    include_eval = session_config.get("visualize.include_eval")
    exclude_eval = session_config.get("visualize.exclude_eval")
    if len(include_eval) == 0:
        include_eval = [""]
    if len(exclude_eval) == 0:
        exclude_eval = [str(random.getrandbits(64))]

    if session_config.get("visualize.module") == "visdom":
        global visdom
        import visdom
        VisdomHandler.run_server(session_config)
    elif session_config.get("visualize.module") == "tensorboard":
        global tensorboard
        from torch.utils import tensorboard
        TensorboardHandler.run_server(session_config)
        session_config.check("visualize.broadcast.enable", [False])
    if command == "visualize":
        VisualizationHandler.post_process_jobs(
            include_train=include_train,
            include_eval=include_eval,
            exclude_train=exclude_train,
            exclude_eval=exclude_eval,
            module=session_config.get("visualize.module"),
            session_type="post",
            folders=session_config.get("visualize.post.folders"),
            embeddings=session_config.get("visualize.post.embeddings")
        )
    else:
        VisualizationHandler.register_broadcast(
            module=session_config.get("visualize.module"),
            include_train=include_train,
            include_eval=include_eval,
            exclude_train=exclude_train,
            exclude_eval=exclude_eval,
            session_type="broadcast"
        )
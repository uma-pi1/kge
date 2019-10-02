from kge.job.trace import Trace
from kge.job import Job, TrainingJob, SearchJob, Trace
from kge.util.misc import kge_base_dir
import os
import yaml
import re
import json
import copy
from collections import defaultdict
import numpy as np
import visdom

class VisualizationHandler():
    """ Base class for broadcasting and post-processing base functionalities for visdom and tensorboard.

    Subclasses implement create(), _visualize_train_item() _visualize_eval_item(), _process_search_trace_entry(),
    post_process_trace()
     
     :param type: "jobtrain", "search", "eval
     :param tracking: "broadcast" or "post" (for post processing the tracefiles..)
     """

    #TODO refactor "tracking" to maybe "session_type"
    def __init__(self,
                 writer,
                 path=None,
                 tracking=None,
                 include_train=None,
                 include_eval=None,
                 jobconfig=None,
                 session_data={}):
        self.writer = writer
        self.tracking = tracking
        self.include_train = include_train
        self.include_eval = include_eval
        self.jobconfig = jobconfig
        # session data can be used to cache any kind of data that is needed during a visualization session
        # during broadcasting this can be best valid.metric during post processing this can be metadata for envs etc.
        self.session_data = session_data
        self.path = path

    @classmethod
    def create_handler(cls, **kwargs):
        if kwargs["module"] == "visdom":
            return VisdomHandler.create(**kwargs)

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    def _process_search_trace_entry(self, trace_entry):
        raise NotImplementedError

    def post_process_trace(self, tracefile, tracetype, jobtype):
        raise NotImplementedError

    def _visualize_config(self):
        raise NotImplementedError

    def _add_config(self):
        """ Loads config."""
        config = self.path + "/" + "config.yaml"
        if os.path.isfile(config):
            with open(config, "r") as file:
                self.config = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            raise Exception("Could not find config.yaml in path.")

    def process_trace(self, tracefile, tracetype, jobtype):
        """ Takes a trace file and processes it.

        :param tracefile:
        :param tracetype: "search", "train", "eval" the type of the trace, this is independent of jobtype because the
        overall jobtype can be e. g. search which also has train type trace files.
        :param jobtype "search", "train", "eval"
        """

        # group traces into dics with {key: list[values]}
        # grouped_entries contains for every distinct trace entry type (train,eval, metadata..)
        # a dic where the keys are the original keys and the values are lists of the original values
        # with that for every plot, server interaction only takes place once
        grouped_entries = []

        with open(tracefile, "r") as file:
            raw = file.readline()
            while(raw):
                trace_entry = yaml.safe_load(raw)
                group_entry = defaultdict(list)
                if len(grouped_entries) == 0:
                    for k,v in trace_entry.items():
                        group_entry[k].append(v)
                    grouped_entries.append(group_entry)
                else:
                    matched = False
                    for entry in grouped_entries:
                        if entry.keys() == trace_entry.keys():
                            for k,v in trace_entry.items():
                                entry[k].append(v)
                            matched = True
                            break
                    if matched == False:
                        for k, v in trace_entry.items():
                            group_entry[k].append(v)
                        grouped_entries.append(group_entry)
                raw = file.readline()
        for entry in grouped_entries:
            self._process_trace_entry(entry, tracetype, jobtype)


    def _process_trace_entry(self, trace_entry, tracetype, jobtype):
        """ Process some trace entry.

       Note that there are different settings which can occur: E. g. a searchjob can have training traces which also have
       eval entries. This has to be tracked because then depending on "broadcast" or "post" behavior differs in the
       various settings.

        """
        entry_keys = list(trace_entry.keys())
        epoch = trace_entry.get("epoch")
        if epoch:
            entry_type = None
            scope = None
            if type(trace_entry["job"]) == list:
                entry_type = trace_entry["job"][0]
                assert(sum(np.array(trace_entry["job"]) != entry_type) == 0)
                scope = trace_entry["scope"][0]
                assert(sum(np.array(trace_entry["scope"]) != scope) == 0)
            else:
                entry_type = trace_entry["job"]
                scope = trace_entry["scope"]
            # catch and ignore 'example' or 'batch' scopes for now
            if scope == "example" or scope == "batch":
                return
            include_patterns = []
            if entry_type == "train":
                include_patterns = self.include_train
                visualize = self._visualize_train_item
            elif entry_type == "eval" and tracetype == "train":
                include_patterns = self.include_eval
                visualize = self._visualize_eval_item
            elif tracetype == "search":
                self._process_search_trace_entry(trace_entry)
                return
            for pattern in include_patterns:
                for matched in list(filter(lambda match_key: re.search(pattern, match_key), entry_keys)):
                    visualize(matched, trace_entry[matched], epoch, tracetype, jobtype)

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
                    tracking=kwargs["tracking"],
                    include_train=kwargs["include_train"],
                    include_eval=kwargs["include_eval"],
                    path=jobpath,
                    session_data=session_data,
                    config=job.config.options
                )
                handler._visualize_config(env=handler.get_env_from_path(tracetype,jobtype))
                def visualize_data(job, trace_entry):
                    handler._process_trace_entry(trace_entry, tracetype=tracetype, jobtype=jobtype)
                job.post_epoch_hooks.append(visualize_data)
                job.valid_job.post_valid_hooks.append(visualize_data)

        Job.job_created_hooks.append(init_hooks)

    @classmethod
    def post_process_jobs(cls, **kwargs):
        """ Scans all the executed jobs in local/experiments and allows submodule to deal with them as they please. """

        path = kge_base_dir() + "/local/experiments/"

        handler = VisualizationHandler.create_handler(
            module=kwargs["module"],
            tracking="post",
            include_train=kwargs["include_train"],
            include_eval=kwargs["include_eval"],
        )
        for parent_job in os.listdir(path):
            parent_trace = path + parent_job + "/trace.yaml"
            first_entry = None
            config = path + parent_job + "/config.yaml"
            if os.path.isfile(parent_trace) and os.path.isfile(config):
                with open(config, "r") as conf:
                    config = yaml.load(conf, Loader=yaml.SafeLoader)
                # pure training job
                if config["job"]["type"] == "train":
                    handler.path = path + parent_job
                    handler.post_process_trace(parent_trace, "train", "train")
                elif config["job"]["type"] == "search":
                    subjobs = []
                    # go through the files in the parent job folder and collect the child training jobs
                    #TODO use filter here or a list comprehension
                    for file in os.listdir(path + "/" + parent_job):
                        child_folder = path + parent_job + "/" + file
                        if os.path.isdir(child_folder) \
                           and file != "config" \
                           and "trace.yaml" in os.listdir(child_folder):
                                subjobs.append(child_folder)
                    handler.path = path + parent_job
                    handler.post_process_trace(parent_trace, "search", "search", subjobs)

        input("waiting for callback")

class VisdomHandler(VisualizationHandler):
    def __init__(
            self,
            writer,
            tracking=None,
            include_train=None,
            include_eval=None,
            path=None,
            config=None,
            jobconfig=None,
            session_data={}):

        super().__init__(writer, path, tracking, include_train, include_eval, jobconfig, session_data)
        self.config = config

    @classmethod
    def create(cls, **kwargs):
        vis = visdom.Visdom()
        return VisdomHandler(
            vis,
            tracking=kwargs.get("tracking"),
            include_train=kwargs.get("include_train"),
            include_eval=kwargs.get("include_eval"),
            path=kwargs.get("path"),
            config=kwargs.get("config"),
            session_data=kwargs.get("session_data"),
            jobconfig=kwargs.get("jobconfig")
        )

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        env = self.get_env_from_path(tracetype, jobtype)
        if jobtype == "train":
            self._visualize_item(key, value, epoch, env=env)
        elif jobtype == "search":
            self._visualize_item(key, value, epoch, env=env)

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        env = self.get_env_from_path(tracetype, jobtype)
        if jobtype == "train":
            self._visualize_item(key, value, epoch, env=env)
        elif jobtype == "search" and tracetype == "train":
            self._visualize_item(key, value, epoch, env=env)
            if key == self.session_data["valid_metric_name"]:
                if self.tracking == "broadcast":
                    # progress plot
                    self._track_progress(key, value, env)
                # plot of all valid.metrics in summary env
                self._visualize_item(
                    key,
                    value,
                    epoch,
                    env=self.extract_summary_envname(env),
                    name=env.split("_")[-1],
                    title=key + "_all",
                    legend=[env.split("_")[-1]],
                    win = key + "_all"
                )
    def _process_search_trace_entry(self, trace_entry):
        # this only works for grouped trace entries, which is fine as it is only used in post processing
        x = None
        names = None
        if "train" in trace_entry["scope"]:
            valid_metric_name = self.session_data["valid_metric_name"]
            x = (np.array(trace_entry[valid_metric_name]))[np.array(trace_entry["scope"])=="train"]
            names = (np.array(trace_entry["folder"]))[np.array(trace_entry["scope"]) == "train"]
            self.writer.bar(
                X=x,
                env=self.extract_summary_envname(envname=self.get_env_from_path("search","search"),jobtype="search"),
                opts={"legend":list(names), "title": valid_metric_name + "_best"}

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
        if type([x][0]) == list:
            x_ = x
            val = value
        else:
            x_ = [x]
            val = [value]

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
    def _visualize_config(self, env):
        self.writer.text(yaml.dump(self.config).replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"), env=env)

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
            self._add_config()
            self._visualize_config(env=self.get_env_from_path("train", "train"))
            def properties_callback(event):
                if event['event_type'] == 'PropertyUpdate':
                    env = event["eid"]
                    # reset the path because when this function is called it might have changed
                    self.path = path
                    self._add_config()
                    self.process_trace(tracefile, tracetype, jobtype)
            self.writer.register_event_handler(properties_callback, properties_window)
        elif jobtype =="search":
            env = self.extract_summary_envname(self.get_env_from_path("search", "search"), "search")
            properties_window = self.writer.properties(properties, env=env)
            self._add_config()
            self._visualize_config(env)
            def properties_callback(event):
                if event['event_type'] == 'PropertyUpdate':
                    self.path = path
                    self._add_config()
                    self.session_data = {"valid_metric_name": self.config.get("valid")["metric"]}
                    self.process_trace(tracefile, "search", "search")
                    # process training jobs of the search job
                    for subjob in sorted(subjobs):
                        self.path = subjob
                        self._add_config()
                        self._visualize_config(self.get_env_from_path("train", "search"))
                        self.session_data = {"valid_metric_name": self.config.get("valid")["metric"]}
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
        # TODO I initially had "key + "_" + env" but then windows overlapped somehow in compare envs
        #  monitor if having the same window name for a title over environemnts leads to problems, but it should not
        return key

    def _get_opts(self, title, **kwargs):
        opts = {
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
    def run_server(cls):
        import subprocess
        from threading import Event
        from os.path import dirname, abspath
        import sys, time
        import signal
        import atexit
        import socket

        DEFAULT_PORT = 8097 #must not be changend atm
        DEFAULT_HOSTNAME = "http://localhost"

        # clean up server in case it's running
        # otherwise start the server
        try:
            s = socket.socket()
            s.connect((DEFAULT_HOSTNAME.replace("http://", ""), DEFAULT_PORT))
            s.close()
            vis = visdom.Visdom()
            for env in vis.get_env_list():
                vis.delete_env(env)
        except:
            PATH = sys.base_exec_prefix + '/bin/' + 'visdom'
            envpath = dirname(dirname(dirname(abspath(__file__)))) + "/local/visdomenv"
            # run server
            process = subprocess.Popen(
                [PATH + " -env_path=" + envpath + " --hostname=" + DEFAULT_HOSTNAME + " -port=" + DEFAULT_PORT],
                shell=True
            )
            time.sleep(1)
            # kill server when everyone's done
            server_pid = process.pid
            def kill_server():
                if server_pid:
                    os.kill(server_pid, signal.SIGTERM)
            atexit.register(kill_server)


def initialize_visualization(config, command):

    #TODO parameter to be obtained from the config
    vis_config = {
        "include_train" : ["avg_loss", "avg_cost", "avg_penalty", "forward_time", "epoch_time"],
        "include_eval":["rank"],
        "tracking" :"broadcast",
        "module":"visdom"
    }
    if config.get("visualize.module") == "visdom":
        VisdomHandler.run_server()
    else:  # tensorboard
        pass

    if command == "visualize":

        VisualizationHandler.post_process_jobs(
            include_train=vis_config["include_train"],
            include_eval=vis_config["include_eval"],
            module=config.get("visualize.module"),
            tracking="post"
        )
    else:

        VisualizationHandler.register_broadcast(
            module=vis_config["module"],
            include_train=vis_config["include_train"],
            include_eval=vis_config["include_eval"],
            tracking="broadcast"
        )
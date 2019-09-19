from kge.job.trace import Trace
from kge.job import Job,TrainingJob,SearchJob
from kge.util.misc import  kge_base_dir
import os
import yaml
import visdom
import re
import json
import copy



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

    def process_trace(self, tracefile, tracetype, jobtype, stop_at=None):
        """ Takes a trace file and processes it.

        :param tracefile:
        :param tracetype: "search", "train", "eval" the type of the trace, this is independent of jobtype because the
        overall jobtype can be e. g. search which also has train type trace files.
        :param jobtype "search", "train", "eval"
        """
        with open(tracefile, "r") as file:
            raw = file.readline()
            idx = 0
            while(raw):
                trace_entry = yaml.safe_load(raw)
                if stop_at and stop_at == idx:
                    return trace_entry
                self._process_trace_entry(trace_entry, tracetype, jobtype)
                raw = file.readline()

    def _process_trace_entry(self, trace_entry, tracetype, jobtype):
        """ Process some trace entry.

       Note that there are different settings which can occur: E. g. a searchjob can have training traces which also have
       eval entries. This has to be tracked because then depending on "broadcast" or "post" behavior differs in the
       various settings.

        """
        entry_keys = list(trace_entry.keys())
        epoch = trace_entry.get("epoch")
        if epoch:
            include_patterns = []
            if trace_entry["job"] == "train":
                include_patterns = self.include_train
                visualize = self._visualize_train_item
            elif trace_entry["job"] == "eval" and tracetype == "train":
                include_patterns = self.include_eval
                visualize = self._visualize_eval_item
            elif trace_entry["job"] == "search":
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
                    session_data=session_data
                )

                def visualize_data(job, trace_entry):
                    handler.process_trace_entry(trace_entry, tracetype=tracetype, jobtype=jobtype)
                job.post_epoch_hooks.append(visualize_data)
                job.valid_job.post_valid_hooks.append(visualize_data)

        Job.job_created_hooks.append(init_hooks)

    @classmethod
    def post_process_jobs(cls, **kwargs):
        """ Scans all the executed jobs in local/experiments and allows submodule to deal with them as they please. """


        path = kge_base_dir() + "/local/experiments/"

        handler = VisualizationHandler.create_handler(
            module=kwargs["module"],
            tracking=kwargs["tracking"],
            include_train=kwargs["include_train"],
            include_eval=kwargs["include_eval"],
        )

        for parent_job in os.listdir(path):
            parent_trace = path + parent_job + "/trace.yaml"
            first_entry = None
            if os.path.isfile(parent_trace):
                with open(parent_trace, "r") as file:
                    raw = file.readline()
                    first_entry = yaml.safe_load(raw)
                # pure training job
                if first_entry["job"] == "train":
                    handler.path = path + parent_job
                    handler.post_process_trace(parent_trace, "train", "train")

        input("waiting for callback")
            # trace = Trace(parent_trace_path)
            # folder = path + "/" + parent_job
            # envname = proc.extract_envname_from_folder(folder)
            # # folder corresponds to a pure training job
            # if trace.entries[0]["job"] == "train":
            #     vis.env = envname
            #     proc.create_sync_property(
            #         vis=vis,
            #         trace_dir=parent_trace_path,
            #         type="train")
            # # folder is a search job with child training jobs
            # elif trace.entries[0]["job"] == "search":
            #     subjobs = []
            #     # go through the files in the the parent job folder and collect the child training jobs
            #     # TODO maybe better check if a yaml exists first, in case a user puts some extra folder in the dirs
            #     for file in os.listdir(path + "/" + parent_job):
            #         if \
            #                 os.path.isdir(path + "/" + parent_job + "/" + file) \
            #                         and file != "config" \
            #                         and "trace.yaml" in os.listdir(folder + "/" + file):
            #             subjobs.append(file)
            #     vis.env = envname + "_asummary"
            #     proc.create_sync_property(
            #         vis=vis,
            #         trace_dir=parent_trace_path,
            #         type="search",
            #         subenvs=[envname + "_" + subjob for subjob in subjobs],
            #         subtraces=[folder + "/" + subjob + "/" + "trace.yaml" for subjob in subjobs]
            #     )
            # proc.best_train_valid = None
            # proc.best_search_valid = None
            # proc.best_train_valid_names = None



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
                self._track_progress(key, value, env)
                self._visualize_item(
                    key,
                    value,
                    epoch,
                    env=self.extract_summary_envname(env),
                    name=env.split("_")[-1],
                )

    def _process_search_trace_entry(self, trace_entry):
        raise NotImplementedError

    def _visualize_item(self, key, value, x, env, name=None , win=None, update="append", **kwargs):

        if win == None:
            win = self.extract_window_name(env, key)
        if name == None:
            name = env.split("_")[-1]

        if not self.writer.win_exists(win, env):
            self.writer.env = env
            self.writer.line(
                X=[x],
                Y=[value],
                win=win,
                opts=self._get_opts(title=key, **kwargs),
                name=name,
            )
        else:
            self.writer.env = env
            self.writer.line(
                X=[x],
                Y=[value],
                win=win,
                opts=self._get_opts(title=key, **kwargs),
                name=name,
                update=update
            )

    def _track_progress(self, key, value, env):
        """ Updates the overall progress plot of a key over multiple train jobs in a search job."""

        # Value is updated whenever a higher value than the current value is found.
        # This is used for valid.metric but can potentially also be used for other keys.
        env = self.extract_summary_envname(env)
        title = "progress_best_{}".format(key)
        win = title + "_" + env
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

    def post_process_trace(self, tracefile, tracetype, jobtype, **kwargs):
        """ Creates an empty environment with a properties button 'sync' which loads the data in the environment. """

        properties = [
            {'type': 'button', 'name': 'Click to sync env', 'value': 'Synchronize'}
        ]
        self.writer.env = self.get_env_from_path(tracetype,jobtype)
        # this returns just the string id of the window
        properties_window = self.writer.properties(properties)
        path = copy.deepcopy(self.path)
        if tracetype == "train":
            def properties_callback(event):
                if event['event_type'] == 'PropertyUpdate':
                    env = event["eid"]
                    # reset the path because when this function is called it might have changed
                    self.path = path
                    self.process_trace(tracefile, tracetype, jobtype)
        self.writer.register_event_handler(properties_callback, properties_window)

    def extract_summary_envname(self, envname):
        """ For an envname of a training job who is part of a searchjob, extracts the summary environment name."""
        env = envname.split("_")[-2]
        env = env + "_" + "SummaryEnvironment"
        return env

    def get_env_from_path(self, tracetype, jobtype ):
        path = self.path.replace("_", "-")
        parts = path.split("/")
        if tracetype == "train" and jobtype == "search":
            return parts[-2] + "_" + parts[-1]
        elif tracetype == "train" and jobtype == "train":
            return parts[-1]

    def extract_window_name(self, env, key):
        return key + "_" + env

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

def initialize_visualization(vis_config):

    #TODO parmeter to be obtained from the vis_config
    vis_config = {
        "include_train" : ["avg_loss", "avg_cost", "avg_penalty", "forward_time", "epoch_time"],
        "include_eval":["rank"],
        "tracking" :"post",
        "module":"visdom"
    }

    if vis_config["tracking"] == "broadcast":

        VisualizationHandler.register_broadcast(
            module=vis_config["module"],
            include_train = vis_config["include_train"],
            include_eval = vis_config["include_eval"],
            tracking = "broadcast"
        )
    elif vis_config["tracking"] == "post":
        VisualizationHandler.post_process_jobs(
            include_train = vis_config["include_train"],
            include_eval = vis_config["include_eval"],
            module= vis_config["module"],
            tracking="post"
        )

def run_server():
    import subprocess
    from threading import Event
    from os.path import dirname, abspath
    import sys, time
    import argparse
    import visdom

    PATH = sys.base_exec_prefix + '/bin/' + 'visdom'
    envpath = dirname(dirname(dirname(abspath(__file__)))) + "/local/visdomenv"
    process = subprocess.Popen([PATH + " -env_path=" + envpath], shell=True)
    time.sleep(5)

    # DEFAULT_PORT = 8080
    # DEFAULT_HOSTNAME = "http://localhost"
    # parser = argparse.ArgumentParser(description='Demo arguments')
    # parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
    #                     help='port the visdom server is running on.')
    # parser.add_argument('-server', metavar='server', type=str,
    #                     default=DEFAULT_HOSTNAME,
    #                     help='Server address of the target to run the demo on.')
    # parser.add_argument('-base_url', metavar='base_url', type=str,
    #                     default='/',
    #                     help='Base Url.')
    # parser.add_argument('-username', metavar='username', type=str,
    #                     default='',
    #                     help='username.')
    # parser.add_argument('-password', metavar='password', type=str,
    #                     default='',
    #                     help='password.')
    # parser.add_argument('-use_incoming_socket', metavar='use_incoming_socket', type=bool,
    #                     default=True,
    #                     help='use_incoming_socket.')
    # FLAGS = parser.parse_args()
    try:
        Event().wait()
    except:
        process.kill()

if __name__ == "__main__":
    run_server()








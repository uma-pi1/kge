from kge.job.trace import Trace
from kge.job import Job,TrainingJob,SearchJob
import yaml
import visdom
import re
import json



class VisualizationHandler():
    """ Base class for broadcasting and post-processing base functionalities for visdom and tensorboard.
     
     :param type: "jobtrain", "search", "eval
     :param tracking: "broadcast" or "post" (for post processing the tracefiles..)
     """

    #TODO refactor "tracking" to maybe "session_type"
    def __init__(self, writer, tracking=None, include_train=None, include_eval=None, jobconfig=None, session_data={}):
        self.writer = writer
        self.tracking = tracking
        self.include_train = include_train
        self.include_eval = include_eval
        self.jobconfig = jobconfig
        # session data can be used to cache any kind of data that is needed during a visualization session
        # during broadcasting this can be best valid.metric during post processing this can be metadata for envs etc.
        self.session_data = session_data

    def process_trace(self, tracefile, tracetype, jobtype):
        """ Takes a trace file and processes it.
        :param tracefile:
        :param tracetype: "search", "train", "eval" the type of the trace, this is independent of jobtype because the
        overall jobtype can be e. g. search which also has train type trace files.
        :param jobtype "search", "train", "eval"
        """
        with open(tracefile, "r") as file:
            raw = file.readline()
            while(raw):
                trace_entry = yaml.safe_load(raw)
                self._process_trace_entry(trace_entry, tracetype, jobtype)
                raw = file.readline()

    def process_trace_entry(self, trace_entry, tracetype, jobtype):
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

    def _process_search_trace_entry(self, trace_entry):
        raise NotImplementedError

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    @classmethod
    def register_broadcast(cls, **kwargs):
        """ Bundles the different information that are needed to perform broadcasting and registers hooks.

        The user parameter inputs are collected and depending on the jobtype, broadcasting functionality is
        registered as hooks.

        """
        def init_hooks(job):

            if isinstance(job, TrainingJob):
                # TODO for tensorboard there are not really envnames only jobpath is needed, so make this more generic
                # TODO and then decision has to be made where to store the tensorboard event file

                #TODO its probably better to give the base class only the "path" and let a base class
                # have its own create method which takes jobtype and then creates itself with all the functionality needed
                jobpath = str(job.config.folder).replace("_", "-")
                jobpath_parts = jobpath.split("/")

                tracetype = None
                jobtype = None
                # pure training job
                if job.parent_job == None:
                    tracetype = "train"
                    jobtype = "train"
                    envname = jobpath_parts[-1]
                    session_data = {}
               # some search job
                if isinstance(job.parent_job, SearchJob):
                    tracetype = "train"
                    jobtype = "search"
                    envname = "_".join([jobpath_parts[-2],jobpath_parts[-1]])
                    session_data = {"valid_metric_name":job.config.get("valid.metric")}

                handler = VisualizationHandler.create(
                    module=kwargs["module"],
                    tracking=kwargs["tracking"],
                    include_train=kwargs["include_train"],
                    include_eval=kwargs["include_eval"],
                    env=envname,
                    session_data=session_data
                )

                def visualize_data(job, trace_entry):
                    handler.process_trace_entry(trace_entry, tracetype=tracetype, jobtype=jobtype)
                job.post_epoch_hooks.append(visualize_data)
                job.valid_job.post_valid_hooks.append(visualize_data)

        Job.job_created_hooks.append(init_hooks)

    @classmethod
    def create(cls, **kwargs):
        if kwargs["module"] == "visdom":
            vis = visdom.Visdom(env=kwargs["env"])
            return VisdomHandler(
                vis,
                tracking = kwargs["tracking"],
                include_train=kwargs["include_train"],
                include_eval=kwargs["include_eval"],
                envname=kwargs["env"],
                session_data=kwargs["session_data"]
            )

class VisdomHandler(VisualizationHandler):
    def __init__(
            self,
            writer,
            tracking=None,
            include_train=None,
            include_eval=None,
            envname=None,
            config=None,
            jobconfig=None,
            session_data={}):

        super().__init__(writer, tracking, include_train, include_eval, jobconfig, session_data)
        self.envname = envname
        self.config = config

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        if jobtype == "train":
            self._visualize_item(key, value, epoch)
        elif jobtype == "search":
            self._visualize_item(key, value, epoch)

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        if jobtype == "train":
            self._visualize_item(key, value, epoch)
        elif jobtype == "search":
            self._visualize_item(key, value, epoch)
            if key == self.session_data["valid_metric_name"]:
                self._track_progress(key, value)

    def _visualize_item(self, key, value, x, env=None, update="append", win=None):
        if env == None:
            env = self.envname
        self.writer.env = env
        if win == None:
            win = key + "_" + self.envname
        # TODO: the create is here when you change the env of your writer, if this env has not been created before
        #  then the update=append will not create a window because it seemingly does not create an env, nothing happens
        # therefore you have to leave out the update command, maybe this bug will be resolved otherwise wrap this function maybe
        if update == "create":
            self.writer.env = env
            self.writer.line(
                X=[x],
                Y=[value],
                win=win,
                opts=self._get_opts(title=key),
                name=env.split("_")[-1],
            )
        else:
            self.writer.env = env
            self.writer.line(
                X=[x],
                Y=[value],
                win=win,
                opts=self._get_opts(title=key),
                name=env.split("_")[-1],
                update=update
            )
        self.writer.env = self.envname

    def _track_progress(self, key, value):
        """ Updates the overall progress plot of a key in a search job over multiple train jobs."""

        # Value is updated whenever a higher value than the current value is found.
        # This is used for valid.metric but could potentially also be used for other keys.
        env = self.envname.split("_")[-2]
        env = env + "_" + "00Summary"
        title = "progress_best_{}".format(key)
        win = title + "_" + env
        check = False
        if self.session_data.get("best_{}".format(key)) and value > self.session_data.get("best_{}".format(key)):
           self.session_data ["best_{}".format(key)] = value
           check = True
        elif not self.session_data.get("best_{}".format(key)):
           self.session_data["best_{}".format(key)] = value
           check = True
        if not self.writer.win_exists(win, env):
            self._visualize_item(title, value, 1, env=env, update="create", win=win)
        elif check == True and (self.writer.win_exists(win,env)):
            data = self.writer.get_window_data(win, env)
            js = json.loads(data)
            best_val = js["content"]["data"][0]["y"][-1]
            step_num = js["content"]["data"][0]["x"][-1]
            if value > best_val:
                self._visualize_item(title, value, step_num+1, env=env, win=win)

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
    if vis_config["tracking"] == "broadcast":

        VisualizationHandler.register_broadcast(
            module=vis_config["module"],
            include_train = ["avg_loss", "avg_cost", "avg_penalty", "forward_time", "epoch_time"],
            include_eval = ["rank"],
            tracking = vis_config["tracking"]
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








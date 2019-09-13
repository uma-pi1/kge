from kge.job.trace import Trace
from kge.job import Job,TrainingJob,SearchJob
import yaml
import visdom



class VisualizationHandler():
    """ Base class for broadcasting and post-processing base functionalities for visdom and tensorboard.
     
     :param type: "jobtrain", "search", "eval
     :param tracking: "broadcast" or "post" (for post processing the tracefiles..)
     """

    def __init__(self, writer, jobtype=None, tracking=None, include_train=None, include_eval=None):
        self.writer = writer
        self.jobtype = type
        self.tracking = tracking
        self.include_train = include_train
        self.include_eval = include_eval

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

        Note there can be "job"=train like entries in a searchtype trace and there can be a traintype trace in a
        search type job. Depending on this configuration and on "broadcast" vs "post" behavior is slightly different.

        """
        entry_keys = list(trace_entry.keys())
        epoch = trace_entry.get("epoch")
        if epoch:
            include_list = []
            if trace_entry["job"] == "train":
                include_list = self.include_train
                visualize = self._visualize_train_item
            elif trace_entry["job"] == "eval" and tracetype != "search":
                include_list = self.include_eval
                visualize = self._visualize_eval_item
            elif trace_entry["job"] == "search":
                self._process_search_trace_entry(trace_entry)
                return
            for key in entry_keys:
                #TODO change for regex here
                if key in include_list:
                    visualize(key, trace_entry[key], epoch, tracetype, jobtype)

    def _process_search_trace_entry(self, trace_entry):
        raise NotImplementedError

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError


class VisdomHandler(VisualizationHandler):
    def __init__(self, writer, include_train, include_eval, main_envname=None, sub_envname=None, type=False, config=None):
        super().__init__(writer, type, include_train=include_train, include_eval=include_eval)
        self.main_envname = main_envname
        self.sub_envname = sub_envname
        self.config = config

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        if jobtype == "train":
            self._visualize_item(key, value, epoch, tracetype, jobtype)

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        if jobtype == "train":
            self._visualize_item(key, value, epoch, tracetype, jobtype)

    def _visualize_item(self, key, value, epoch, tracetype, jobtype):
        env = None
        if self.sub_envname:
            env =  self.sub_envname
        else:
            env = self.main_envname

        self.writer.env = env
        self.writer.line(
            X=[epoch],
            Y=[value],
            win=key,
            opts=self._get_opts(title=key),
            name=env.split("_")[-1],
            update="append"
        )

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





def initialize(job):

    if isinstance(job, TrainingJob) and job.parent_job == None:
        envname = str(job.config.folder).replace("_", "-")
        envname = envname.split("/")
        train_envname = envname[-1]
        vis = visdom.Visdom(env=train_envname)

        include_train = ["avg_loss", "avg_cost", "avg_penalty", "forward_time", "epoch_time"]
        include_eval = ["mean_rank_filtered"]
        handler = VisdomHandler(vis, main_envname=train_envname, include_train=include_train, include_eval=include_eval)

        def hook1(job, trace_entry):
            handler.process_trace_entry(trace_entry, tracetype="train", jobtype="train")

        job.post_epoch_hooks.append(hook1)
        job.valid_job.post_valid_hooks.append(hook1)




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








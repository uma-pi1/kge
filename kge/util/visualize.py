from kge.job.trace import Trace
from kge.job import Job,TrainingJob,SearchJob
import yaml
import visdom
import re



class VisualizationHandler():
    """ Base class for broadcasting and post-processing base functionalities for visdom and tensorboard.
     
     :param type: "jobtrain", "search", "eval
     :param tracking: "broadcast" or "post" (for post processing the tracefiles..)
     """

    def __init__(self, writer, tracking=None, include_train=None, include_eval=None, jobconfig=None):
        self.writer = writer
        self.tracking = tracking
        self.include_train = include_train
        self.include_eval = include_eval
        self.jobconfig = jobconfig

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
            elif trace_entry["job"] == "eval" and tracetype != "search":
                include_patterns = self.include_eval
                visualize = self._visualize_eval_item
            elif trace_entry["job"] == "search":
                self._process_search_trace_entry(trace_entry)
                return
            for pattern in include_patterns:
                for matched_key in list(filter(lambda key: re.match(pattern, key), entry_keys)):
                    visualize(matched_key, trace_entry[matched_key], epoch, tracetype, jobtype)

    def _process_search_trace_entry(self, trace_entry):
        raise NotImplementedError

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        raise NotImplementedError

    @classmethod
    def register_broadcast(cls, **kwargs):
        """ Bundles the different information that are needed to process broadcasting and registers hooks.

        The user parameter inputs are collected and depending on the jobtype, broadcasting functionality is
        registered as hooks.

        """

        def init_hooks(job):

            if isinstance(job, TrainingJob):
                # TODO for tensorboard this probably has to be changed to path or so
                # TODO and then decision has to be made where to store the tensorboard event file
                envname = str(job.config.folder).replace("_", "-")
                envname = envname.split("/")
                envname = envname[-1]

                handler = VisualizationHandler.create(
                    module=kwargs["module"],
                    tracking=kwargs["tracking"],
                    include_train=kwargs["include_train"],
                    include_eval=kwargs["include_eval"],
                    env=envname
                )
                tracetype = None
                jobtype = None
                # pure training job
                if job.parent_job == None:
                    tracetype = "train"
                    jobtype = "train"
               # some search job
                if isinstance(job.parent_job, SearchJob):
                    tracetype = "train"
                    jobtype = "search"

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
            jobconfig=None):

        super().__init__(writer, tracking, include_train, include_eval, jobconfig)
        self.envname = envname
        self.config = config

    def _visualize_item(self, key, value, epoch, tracetype, jobtype):
        env = self.envname
        self.writer.env = env
        self.writer.line(
            X=[epoch],
            Y=[value],
            win=key,
            opts=self._get_opts(title=key),
            name=env.split("_")[-1],
            update="append"
        )

    def _visualize_train_item(self, key, value, epoch, tracetype, jobtype):
        if jobtype == "train":
            self._visualize_item(key, value, epoch, tracetype, jobtype)
        if jobtype == "search":
            self._visualize_item(key, value, epoch, tracetype, jobtype)


    def _visualize_eval_item(self, key, value, epoch, tracetype, jobtype):
        if jobtype == "train":
            self._visualize_item(key, value, epoch, tracetype, jobtype)

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
            include_eval = ["mean_rank_filtered"],
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








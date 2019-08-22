import visdom
import yaml
from kge.util.vis_support import VisdomBroadcastHandler
from os.path import dirname,abspath


class VisdomPostProcessor(VisdomBroadcastHandler):
    """ Handles post processing of trace files for the integration of Visdom features. """

    #TODO: much of the functionality is similar to the broadcaster module but in there we use
    # jobs of the framework and e. g. also the hooks depend on job's and Visdom objects that are defined
    # out of the scope of the function and cannot be passed as parameters. In the hooks, I'm accessing self.job all the time
    # but this here is post processing, I don't have a job object here. Maybe you could create a job with the config
    # then you can just access the needed items


    def __init__(self, include_train, include_eval):

        # TODO make this user definable you could e. g. run the post process part also easily with some
        # TODO kge like config
        assert(len(include_eval) > 0)
        assert(len(include_train) > 0)
        self.include_train = include_train
        self.include_eval = include_eval
        self.search_valids = []

    def _parse_training_trace_entries(self, trace_entry, vis):
        """ Takes an entry of a training trace and handels visualizations."""
        #TODO: this is now sort of visdom_trackig_training_metrics hook together with tracking_eval hook

        epoch = None
        if "epoch" in list(trace_entry.keys()):
            epoch = trace_entry["epoch"]
            # track train metrics
            if trace_entry["job"] == "train":
                for track_me in self.include_train:
                    # create or update plot for train metrics
                    self._vis_line(
                        vis=vis,
                        X=[epoch],
                        Y=[trace_entry[track_me]],
                        win=track_me,
                        opts=self._get_opts(track_me),
                )
            # track eval metrics
            # in the trainin job trace are also entries of the child-valid job
            elif trace_entry["job"] == "eval":
                for metric in trace_entry.keys():
                    for track_me in self.include_eval:
                        if track_me in metric:
                            # create or update plot for valid metrics
                            self._vis_line(
                                vis=vis,
                                X=[epoch],
                                Y=[trace_entry[metric]],
                                win=metric,
                                opts=self._get_opts(title=metric)
                            )

    def parse_trace(self, trace_dir, func, vis):
        """ Takes the path of a trace file and a function and creates data in an environment. """
        trace = None
        path = trace_dir + "/trace.yaml"
        with open(path, "r") as file:
            txt = file.readline()
            while(txt):
                txt = file.readline()
                trace_entry = yaml.safe_load(txt)
                # TODO hacky here, this prevents an error at the end of the file
                if trace_entry:
                    func(trace_entry, vis)

    def _get_opts(self, title):
        # cannot use the base version atm because job is used
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

def main():
    path = dirname(dirname(dirname(abspath(__file__)))) + "/local/experiments"
    trace_dir = "/home/patrick/Desktop/kge/local/experiments/20190821-221408-toy-complex-grid/dim20_lr0.01"
    include_train = ["avg_loss", "avg_cost", "avg_penalty","forward_time","epoch_time"]
    include_eval = ["rank"]
    proc = VisdomPostProcessor(include_train, include_eval)
    vis = visdom.Visdom(env="Env")
    proc.parse_trace(trace_dir, proc._parse_training_trace_entries,vis)
if __name__== "__main__":
    main()












from typing import List
import torch
from kge import Config
from kge.job import SearchJob

# TODO handle "max_epochs" in some sensible way


class AutoSearchJob(SearchJob):
    """Base class for search jobs that automatically explore the search space.

    Subclasses should implement :func:`init_search`, :func:`register_trial`,
    :func:`register_trial_result`, :func:`get_best_parameters`.

    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        self.trial_ids: List = []  #: backend-specific identifiers for each trial
        self.parameters: List[dict] = []  #: hyper-parameters of each trial
        self.results: List[dict] = []  #: trace entry of best result of each trial

    def load(self, filename):
        self.config.log("Loading checkpoint from {}...".format(filename))
        checkpoint = torch.load(filename)
        self.parameters = checkpoint["parameters"]
        self.results = checkpoint["results"]

    def save(self, filename):
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save({"parameters": self.parameters, "results": self.results}, filename)

    def resume(self):
        last_checkpoint = self.config.last_checkpoint()
        if last_checkpoint is not None:
            checkpoint_file = self.config.checkpoint_file(last_checkpoint)
            self.load(checkpoint_file)
        else:
            self.config.log("No checkpoint found, starting from scratch...")

    # -- Abstract methods --------------------------------------------------------------

    def init_search(self):
        """Initialize to start a new search experiment."""
        raise NotImplementedError

    def register_trial(self, parameters=None):
        """Start a new trial.

        If parameters is ``None``, automatically determine a suitable set of parameter
        values. Otherwise, use the specified parameters.

        Return a (parameters, trial id)-tuple or ``(None, None)``. The trial id is not
        further interpreted, but can be used to store metadata for this trial.
        ``(None,None)`` is returned only if parameters is ``None`` and there are

        registered trials for which the result has not yet been registered via
        :func:`register_trial_results`.

        """
        raise NotImplementedError

    def register_trial_result(self, trial_id, parameters, trace_entry):
        """Register the result of a trial.

        `trial_id` and `parameters` should have been produced by :func:`register_trial`.
        `trace_entry` should be the trace entry for the best validation result of a
        training job with these parameters.

        """
        raise NotImplementedError

    def get_best_parameters(self):
        """"Return a (best parameters, estimated objective value) tuple."""
        raise NotImplementedError

    def run(self):
        """Runs the HPO algorithm."""
        raise NotImplementedError

    # -- Main --------------------------------------------------------------------------

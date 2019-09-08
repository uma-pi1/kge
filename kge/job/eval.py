import torch

from kge.job import Job


class EvaluationJob(Job):
    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job)

        self.config = config
        self.dataset = dataset
        self.model = model
        self.batch_size = config.get("eval.batch_size")
        self.device = self.config.get("job.device")
        max_k = min(self.dataset.num_entities, max(self.config.get("eval.hits_at_k_s")))
        self.hits_at_k_s = list(
            filter(lambda x: x <= max_k, self.config.get("eval.hits_at_k_s"))
        )
        self.config.check("train.trace_level", ["example", "batch", "epoch"])
        self.trace_examples = self.config.get("eval.trace_level") == "example"
        self.trace_batch = (
            self.trace_examples or self.config.get("train.trace_level") == "batch"
        )
        self.eval_data = self.config.check("eval.data", ["valid", "test"])
        self.filter_valid_with_test = config.get("valid.filter_with_test")
        self.epoch = -1

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks = []

        #: Hooks run after a validation job.
        #: Signature: job, trace_entry
        self.post_valid_hooks = []

        #: Hooks after computing the metrics and compute some specific metrics .
        #: Signature: job, trace_entry
        self.hist_hooks = []

        self.hist_hooks.append(KeepAllEvaluationHistogramHooks())

        if config.get("eval.metric_per_relation_type"):
            self.dataset.load_relation_types()
            self.hist_hooks.append(RelationTypeEvaluationHistogramHooks())

    @staticmethod
    def create(config, dataset, parent_job=None, model=None):
        """Factory method to create an evaluation job """
        from kge.job import EntityRankingJob, EntityPairRankingJob

        # create the job
        if config.get("eval.type") == "entity_ranking":
            return EntityRankingJob(config, dataset, parent_job=parent_job, model=model)
        elif config.get("eval.type") == "entity_pair_ranking":
            return EntityPairRankingJob(
                config, dataset, parent_job=parent_job, model=model
            )
        else:
            raise ValueError("eval.type")

    def run(self) -> dict:
        """ Compute evaluation metrics, output results to trace file """
        raise NotImplementedError

    def resume(self):
        # load model
        from kge.job import TrainingJob

        training_job = TrainingJob.create(self.config, self.dataset)
        training_job.resume()
        self.model = training_job.model
        self.epoch = training_job.epoch
        self.resumed_from_job = training_job.resumed_from_job


class EvaluationHistogramHooks:
    """
    Filters the batch rank based on the entities and relations.
    """
    def init_hist_hook(self, eval_job, dataset, device, dtype):
        raise NotImplementedError

    def make_batch_hist(self, hist, dataset, s, p, o, s_ranks, o_ranks, device, dtype=torch.float):
        raise NotImplementedError


class KeepAllEvaluationHistogramHooks(EvaluationHistogramHooks):
    """
    Default class that does nothing.
    """
    def init_hist_hook(self, eval_job, dataset, device, dtype):
        return {'all': torch.zeros([dataset.num_entities], device=device, dtype=dtype)}

    def make_batch_hist(self, hist_dict, dataset, s, p, o, s_ranks, o_ranks, device, dtype=torch.float):
        num_entities = dataset.num_entities
        # need for loop because batch_hist[o_ranks]+=1 ignores repeated
        # entries in o_ranks
        for r in o_ranks:
            hist_dict['all'][r] += 1
        for r in s_ranks:
            hist_dict['all'][r] += 1
        return hist_dict

class RelationTypeEvaluationHistogramHooks(EvaluationHistogramHooks):
    """
    Filters the batch rank by relation type.
    """
    def init_hist_hook(self, eval_job, dataset, device, dtype):
        result = dict()
        for rtype in dataset.relations_per_type.keys():
            result[rtype] = torch.zeros([dataset.num_entities], device=device, dtype=dtype)
        return result

    def make_batch_hist(self, hist_dict, dataset, s, p, o, s_ranks, o_ranks, device, dtype=torch.float):
        masks = list()
        for rtype in dataset.relations_per_type.keys():
            masks.append(
                (rtype, [_p in dataset.relations_per_type[rtype] for _p in p.tolist()])
            )
        for (rtype, mask) in masks:
            for r, m in zip(o_ranks, mask):
                if m: hist_dict[rtype][r] += 1
            for r, m in zip(s_ranks, mask):
                if m: hist_dict[rtype][r] += 1
        return hist_dict

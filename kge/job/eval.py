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
        self.predict_output = config.get("eval.predict_output")
        self.predict_output_k = config.get("eval.predict_output_k")
        self.predict_output_log = config.get("eval.predict_output_log")

        #: Hooks run after evaluation for an epoch.
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

        #: Signature: job, trace_entry
        self.post_valid_hooks = []

        #: Hooks after computing the ranks for each batch entry.
        #: Signature: job, trace_entry
        self.hist_hooks = [hist_all]
        if config.get("eval.metrics_per.head_and_tail"):
            self.hist_hooks.append(hist_per_head_and_tail)
        if config.get("eval.metrics_per.relation_type"):
            self.hist_hooks.append(hist_per_relation_type)
        if config.get("eval.metrics_per.argument_frequency"):
            self.hist_hooks.append(hist_per_frequency_percentile)

        # all done, run job_created_hooks if necessary
        if self.__class__ == EvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

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

    def resume(self, checkpoint_file=None):
        """Load model state from last or specified checkpoint."""
        # load model
        from kge.job import TrainingJob

        training_job = TrainingJob.create(self.config, self.dataset)
        training_job.resume(checkpoint_file)
        self.model = training_job.model
        self.epoch = training_job.epoch
        self.resumed_from_job_id = training_job.resumed_from_job_id


## HISTOGRAM COMPUTATION ################################################################

def __initialize_hist(hists, key, job):
    """If there is no histogram with given `key` in `hists`, add an empty one."""
    if key not in hists:
        hists[key] = torch.zeros(
            [job.dataset.num_entities],
            device=job.config.get("job.device"),
            dtype=torch.float,
        )


def hist_all(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    """Create histogram of all subject/object ranks (key: "all").

    `hists` a dictionary of histograms to update; only key "all" will be affected. `s`,
    `p`, `o` are true triples indexes for the batch. `s_ranks` and `o_ranks` are the
    rank of the true answer for (?,p,o) and (s,p,?) obtained from a model.

    """
    __initialize_hist(hists, "all", job)
    hist = hists["all"]
    for r in o_ranks:
        hist[r] += 1
    for r in s_ranks:
        hist[r] += 1


def hist_per_head_and_tail(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    __initialize_hist(hists, "head", job)
    hist = hists["head"]
    for r in s_ranks:
        hist[r] += 1

    __initialize_hist(hists, "tail", job)
    hist = hists["tail"]
    for r in o_ranks:
        hist[r] += 1


def hist_per_relation_type(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    job.dataset.index_relation_types()
    masks = list()
    for rel_type, rels in job.dataset.indexes["relations_per_type"].items():
        __initialize_hist(hists, rel_type, job)
        mask = [_p in rels for _p in p.tolist()]
        for r, m in zip(o_ranks, mask):
            if m:
                hists[rel_type][r] += 1
        for r, m in zip(s_ranks, mask):
            if m:
                hists[rel_type][r] += 1


def hist_per_frequency_percentile(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    # initialize
    job.dataset.index_frequency_percentiles()
    frequency_percs = job.dataset.indexes["frequency_percentiles"]
    for arg, percs in frequency_percs.items():
        for perc, value in percs.items():
            __initialize_hist(hists, "{}_{}".format(arg, perc), job)

    # go
    for perc in frequency_percs["subject"].keys():  # same for relation and object
        for r, m_s, m_r in zip(
            s_ranks,
            [id in frequency_percs["subject"][perc] for id in s.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_s:
                hists["{}_{}".format("subject", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
        for r, m_o, m_r in zip(
            o_ranks,
            [id in frequency_percs["object"][perc] for id in o.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_o:
                hists["{}_{}".format("object", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1


    def __init__(self, dataset):
        super().__init__(dataset)
        self.frequency_perc = (
            self.dataset.get_frequency_percentiles_for_entites_and_relations()
        )

    def init_hist_hook(self, device, dtype):
        result = dict()
        for arg in self.frequency_perc.keys():
            for perc in self.frequency_perc[arg].keys():
                result["{}_{}".format(arg, perc)] = torch.zeros(
                    [self.dataset.num_entities], device=device, dtype=dtype
                )
        return result

    def make_batch_hist(
        self, hist_dict, s, p, o, s_ranks, o_ranks, device, dtype=torch.float
    ):
        for perc in self.frequency_perc[
            "subject"
        ].keys():  # same for relation and object
            for r, m_s, m_r in zip(
                s_ranks,
                [id in self.frequency_perc["subject"][perc] for id in s.tolist()],
                [id in self.frequency_perc["relation"][perc] for id in p.tolist()],
            ):
                if m_s:
                    hist_dict["{}_{}".format("subject", perc)][r] += 1
                if m_r:
                    hist_dict["{}_{}".format("relation", perc)][r] += 1
            for r, m_o, m_r in zip(
                o_ranks,
                [id in self.frequency_perc["object"][perc] for id in o.tolist()],
                [id in self.frequency_perc["relation"][perc] for id in p.tolist()],
            ):
                if m_o:
                    hist_dict["{}_{}".format("object", perc)][r] += 1
                if m_r:
                    hist_dict["{}_{}".format("relation", perc)][r] += 1

        return hist_dict


class output_predictions_per_triple():

    def __init__(self, config):
        self.config = config

    def _load_map(self, filename):
        import csv
        # Read the entities from file
        # Todo: Move function to dataset.py or change the function in dataset.py for reading the entities.txt file
        #   First, mappings to real names of entities of othewr datasets have to be added
        dictionary = {}
        with open(filename, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                index = row[0]
                meta = row[1:]
                dictionary[index] = meta
        return dictionary

    def get_best_predictions_per_triple(self, triples, scores, k):
        best_predictions_per_triple = {}
        # Loop over every test triple to find the best predictions for it
        for t in range(len(triples)):
            # Get indices of the k largest scores
            indices = scores[t,:].topk(k).indices

            # Get best triples
            best_triples = torch.as_tensor([triples[t].tolist()] * k)

            for i, j in enumerate(indices):
                if j < self.dataset.num_entities:
                    best_triples[:, 0][i] = j
                else:
                    best_triples[:, 2][i] = j - self.dataset.num_entities

            best_predictions_per_triple[triples[t]] = best_triples

        return best_predictions_per_triple

    # Todo: Need a complete and unique-valued, english-only mapping, atm output is in different languages and unreadable

    def create_output_best_predictions(self, best_predictions_per_triple, entities_map):
        # Todo after feedback if approach is good: Make the predictions more beuatiful in trace and output file
        # For printing only last part of relation add: .rsplit('/', 1)[1][:-2]
        predictions = {}
        triple_count = 1
        for i, j in zip(best_predictions_per_triple.keys(), best_predictions_per_triple.values()):
            # At the moment, we don't have a mapping to real names where every entity is listed. Therefore,
            # we need to output the IDs in those cases. Whenever an ID key is not present in the mapping, we use the ID.
            try:
                predictions[str(entities_map[''.join(self.dataset.entities[int(i[0])])]) +
                            str(self.dataset.relations[int(i[1])]) +
                            str(entities_map[''.join(self.dataset.entities[int(i[2])])])] = \
                    ["Best prediction no. {}: ".format(n+1) +
                     str(entities_map[''.join(self.dataset.entities[int(t[0])])] +
                         self.dataset.relations[int(t[1])] +
                    entities_map[''.join(self.dataset.entities[int(t[2])])]) for n,t in enumerate(j)]
            except KeyError:
                predictions[str(entities_map[''.join(self.dataset.entities[int(i[0])])]) +
                            str(self.dataset.relations[int(i[1])]) +
                            str(entities_map[''.join(self.dataset.entities[int(i[2])])])] = \
                    ["Best prediction no. {}: ".format(n+1) +
                     ''.join(self.dataset.entities[int(t[0])]) +
                         str(self.dataset.relations[int(t[1])]) +
                    ''.join(self.dataset.entities[int(t[2])]) for n,t in enumerate(j)]

            triple_count += 1

        return predictions
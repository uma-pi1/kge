import time
import random

import torch
from sklearn.metrics import accuracy_score, precision_score
from kge.job import EvaluationJob


class TripleClassificationJob(EvaluationJob):
    """Triple classification evaluation protocol:
    Testing model's ability to discriminate between true and false triples based on scores. Introduces a treshold for
    each relation. Unseen triples will be predicted as True if the score is higher than the treshold.
    Todo: Get rid of as many for loops as possible to make the evaluation faster!!
    """
    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.threshold_data = self.config.get("eval.thresholds")
        self.eval_data = self.config.get("eval.test") #Todo: Use eval.data and delete eval.test in configuration (didnt work for some reason)
        self.is_prepared = False

    def _prepare(self):
        """Load specified data."""

        if self.is_prepared:
            return

        # Set test dataset
        if self.eval_data == "test":
            self.eval = self.dataset.test
        else:
            self.eval = self.dataset.valid

        # Set dataset for which thresholds are found
        if self.threshold_data == "valid":
            self.threshold = self.dataset.valid
        else:  self.threshold = self.dataset.train

        # let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    def run(self):
        """1. Generation of (corrupted) negative triples:
               Corrupt each triple in valid and test data once to get equally amount of wrong and correct triples.
               Allow only entities which appeared at the given position in the dataset
            2. Get scores for the corrupted datasets
            3. Find the best threshold for every relation by maximizing accuracy on validation data
            4. Classify triples in test data
            5. Compute Metrics for test data
            6. Trace & Log
        """
        self._prepare()

        was_training = self.model.training #Todo-Question: Copied that from entity ranking but don't know if it is needed
        self.model.eval()

        self.config.log("Starting triple classification...")
        epoch_time = -time.time()

        # 1. Generate corrupted data. Output: triples, labels, labels per relation
        self.config.log("Generate corrupted datasets...")
        valid_corrupted, valid_labels, rel_valid_labels = self._generate_negatives(self.threshold)
        test_corrupted, test_labels, rel_test_labels = self._generate_negatives(self.eval)

        # 2. Get scores for the new data. Relevant Output: Scores and scores per relation
        self.config.log("Get scores for datasets...")
        s_valid, p_valid, o_valid = valid_corrupted[:, 0], valid_corrupted[:, 1], valid_corrupted[:, 2]
        valid_scores = self.model.score_spo(s_valid, p_valid, o_valid)
        rel_valid_scores = {int(r): valid_scores[(p_valid == r).nonzero(),:] for r in p_valid.unique()}

        s_test, p_test, o_test = test_corrupted[:, 0], test_corrupted[:, 1], test_corrupted[:, 2]
        test_scores = self.model.score_spo(s_test, p_test, o_test)
        rel_test_scores = {int(r): test_scores[(p_test == r).nonzero(),:] for r in p_test.unique()}

        # 3. Find the best thresholds for every relation and their accuracies on the valid data
        self.config.log("Learning thresholds on " + self.threshold_data +  " data.")
        rel_thresholds, accuracies_valid = self.findThresholds(p_valid, rel_valid_scores, rel_valid_labels)

        # 4. Classification on test data. Output: predictions per relation and number of relations in test which are
        # not included in valid
        self.config.log("Evaluating on " + self.eval_data + " data.")
        self.config.log("Predict...")
        rel_predictions, not_in_eval = self.predict(rel_thresholds, rel_test_scores, p_valid, p_test)

        # 5. Report Metrics on test data
        self.config.log("Classification results:")
        metrics = self._compute_metrics(rel_test_labels, rel_predictions, p_valid, p_test, not_in_eval)

        # 6. Trace & Log

        epoch_time += time.time()
        # compute trace
        trace_entry = dict(
            type="triple_classification",
            scope="epoch",
            data_learn_thresholds=self.threshold_data,
            data_evaluate=self.eval_data,
            epoch=self.epoch,
            size=2*len(self.eval),
            epoch_time=epoch_time,
            **metrics,
        )
        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in trace_entry:
            trace_entry[metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                {"config": self.config, **trace_entry},
            )

        # write out trace
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log("Finished evaluating on " + self.eval_data + " data.")

        return trace_entry
    # Todo-Question: Not sure if what is included in the trace is correct or enough. Feedback needed.

    def _generate_negatives(self, dataset):
        # 1. Corrupt triples
        labels = []
        corrupted = []
        for triple in dataset:
            corrupted.append(triple)
            labels.append(1)
            # Random decision if sample subject(False) or object(True)
            if bool(random.getrandbits(1))==True:
                s = corrupted[-1][0]
                p = corrupted[-1][1]
                o = random.sample(list(dataset[:,2]), 1)[0]
                # Guarantee that s!=o and that the sampled triple is not a true triple of any other dataset
                while int(s)==int(o) \
                        and torch.tensor([s, p, o], dtype=torch.int32) in self.dataset.train\
                        and torch.tensor([s, p, o], dtype=torch.int32) in self.dataset.valid\
                        and torch.tensor([s, p, o], dtype=torch.int32) in self.dataset.test:
                    o = random.sample(list(dataset[:,2]), 1)[0]
            else:
                s = random.sample(list(dataset[:,0]), 1)[0]
                p = corrupted[-1][1]
                o = corrupted[-1][2]
                # Guarantee that s!=o and that the sampled triple is not a true triple of any other dataset
                while int(s) == int(o) \
                        and torch.tensor([s, p, o], dtype=torch.int32) in self.dataset.train \
                        and torch.tensor([s, p, o], dtype=torch.int32) in self.dataset.valid \
                        and torch.tensor([s, p, o], dtype=torch.int32) in self.dataset.test:
                    o = random.sample(list(dataset[:,0]), 1)[0]

            corrupted.append(torch.tensor([s, p, o], dtype=torch.int32))
            labels.append(0)
        corrupted = torch.stack(corrupted)

        # TODO-Question: Would it make sense to use and modify util.sampler for that task?
        # TODO-Question: Right now we allow only samples at the position where they appeared and only from the same dataset as specified.
        #  Would it make sense to allow to sample from all three available datasets?

        # Save the labels per relation, since this will be needed frequently later
        p = corrupted[:, 1]
        rel_labels = {int(r): [labels[int((p == r).nonzero()[i])]
                               for i in range(len((p == r).nonzero()))] for r in p.unique()}

        return corrupted, labels, rel_labels

    def findThresholds(self, p, rel_scores, rel_labels):
        # Initialize accuracies, thresholds (and predictions)
        rel_accuracies = {int(r): -1 for r in p.unique()}
        rel_thresholds = {int(r): 0 for r in p.unique()}
#        rel_predictions = {int(r): 0 for r in p.unique()}

        # Find best thresholds
        for r in p.unique():
            for t in rel_scores[int(r)]:
                preds = torch.zeros(len((p == r).nonzero()))
                for i in range(len(rel_scores[int(r)])):
                    if rel_scores[int(r)][i] >= t:
                        preds[i] = 1
                accuracy = accuracy_score(rel_labels[int(r)], preds)
                if accuracy > rel_accuracies[int(r)]:
                    rel_accuracies[int(r)] = accuracy
                    rel_thresholds[int(r)] = float(t)
                    #rel_predictions[int(r)] = preds

        return rel_thresholds, rel_accuracies

    def predict(self, rel_thresholds, rel_scores, p_valid, p_test):

        rel_predictions = {int(r):[0]*len(rel_scores[int(r)]) for r in p_test.unique()}

        # Set counter for triples for which the relation is not in valid data
        not_in_eval = []
        for r in p_test.unique():
            # Check if relation which is in valid data also is in test data
            if r in p_valid.unique():
                # Predict
                for i in range(len(rel_scores[int(r)])):
                        if float(rel_scores[int(r)][i]) >= rel_thresholds[int(r)]:
                            rel_predictions[int(r)][i] = 1
            else: not_in_eval.append(r)

        return rel_predictions, not_in_eval

    def _compute_metrics(self, rel_test_labels, rel_predictions, p_valid, p_test, not_in_eval):
        metrics = {}

        labels_in_test_list = [i
                     for r in p_test.unique()
                     for i in rel_test_labels[int(r)]]

        pred_list = [i
                     for r in p_test.unique()
                     for i in rel_predictions[int(r)]]


        metrics["Accuracy"] = float(accuracy_score(labels_in_test_list, pred_list))
        metrics["Precision"] = float(precision_score(labels_in_test_list, pred_list))

        precision_per_r = {}
        accuracy_per_r = {}
        for r in p_test.unique():
                precision_per_r[str(self.dataset.relations[int(r)])] = float(precision_score(rel_test_labels[int(r)], rel_predictions[int(r)]))
                accuracy_per_r[str(self.dataset.relations[int(r)])] = float(accuracy_score(rel_test_labels[int(r)], rel_predictions[int(r)]))
        # Todo: Find out what the warning "UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
        #   'precision', 'predicted', average, warn_for)" is.
        metrics["Accuracy_per_Relation"] = accuracy_per_r

        metrics["Precision_Per_Relation"] = precision_per_r

        # Since we evaluate on test data, only the relations in the test data which cannot be evaluated are counted here.
        # In general we miss more than teh half of the existing relations for toy data, because they are not in test/valid.
        metrics["Untested relations due to missing in evaluation data"] = len(not_in_eval)

        return metrics

    # TODO-Question: We optimized the tresholds only for one randomly corrupted sample of the data.
    #  Another sample would give (a little) different results due to a different threshold.
    #  I would probably optimize the thresholds for different samples and in the end take something like the mean of all
    #  thresholds as final threshold, but in the literature, it seems like they really corrupt the data only once.
    #  Anyway for comparison of models, we have to pay attention to use the same data samples.Thus it might be better to
    #  create and save a dataset with negative labels and use always the same for all models.
    #  Any feedback on this?
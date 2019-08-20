import time
import random
import itertools
from copy import deepcopy

import torch
from sklearn.metrics import accuracy_score, precision_score
from kge.job import EvaluationJob


class TripleClassificationJob(EvaluationJob):
    """Triple classification evaluation protocol:
    Testing model's ability to discriminate between true and false triples based on scores. Introduces a treshold for
    each relation. Unseen triples will be predicted as True if the score is higher than the treshold.

    """
    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        # 1. Generate corrupted data. Output: triples, labels, labels per relation
        self.config.log("Generate corrupted datasets...")
        # Create the corrupted triples while creating the evaluation Job to make sure that every epoch is evaluated on the same data
        self.valid_corrupted, self.valid_labels, self.rel_valid_labels = self._generate_negatives(self.dataset.valid)
        self.test_corrupted, self.test_labels, self.rel_test_labels = self._generate_negatives(self.dataset.test)


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

        was_training = self.model.training
        self.model.eval()

        self.config.log("Starting triple classification...")
        epoch_time = -time.time()

        # 1. Generate corrupted data - already done

        # 2. Get scores for the new data. Relevant Output: Scores and scores per relation
        self.config.log("Get scores for datasets...")
        s_valid, p_valid, o_valid = self.valid_corrupted[:, 0], self.valid_corrupted[:, 1], self.valid_corrupted[:, 2]
        valid_scores = self.model.score_spo(s_valid, p_valid, o_valid)
        rel_valid_scores = {int(r): valid_scores[(p_valid == r).nonzero(),:] for r in p_valid.unique()}

        s_test, p_test, o_test = self.test_corrupted[:, 0], self.test_corrupted[:, 1], self.test_corrupted[:, 2]
        test_scores = self.model.score_spo(s_test, p_test, o_test)
        rel_test_scores = {int(r): test_scores[(p_test == r).nonzero(),:] for r in p_test.unique()}

        # 3. Find the best thresholds for every relation and their accuracies on the valid data
        self.config.log("Learning thresholds on validation data.")
        rel_thresholds, accuracies_valid = self.findThresholds(p_valid, valid_scores, rel_valid_scores, self.valid_labels, self.valid_corrupted)

        # 4. Classification on test data. Output: predictions per relation and number of relations in test which are
        # not included in valid
        self.config.log("Evaluating on test data.")
        self.config.log("Predict...")
        rel_predictions, not_in_eval = self.predict(rel_thresholds, test_scores, rel_test_scores, p_valid, p_test)

        # 5. Report Metrics on test data
        self.config.log("Classification results:")
        metrics = self._compute_metrics(self.rel_test_labels, rel_predictions, p_valid, p_test, not_in_eval)

        # 6. Trace & Log

        epoch_time += time.time()
        # compute trace
        trace_entry = dict(
            type="triple_classification",
            scope="epoch",
            data_learn_thresholds="Valid",
            data_evaluate="Test",
            epoch=self.epoch,
            size=2*len(self.dataset.valid),
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
        corrupted = dataset.repeat(1, 2).view(-1, 3)
        labels = torch.as_tensor([1, 0] * len(dataset))

        sample = torch.randint(0,2,(1,len(dataset)))

        # Random decision if sample subject(sample=nonzero) or object(sample=zero)
        corrupted[1::2][:, 0][sample.nonzero()[:, 1]] = \
            torch.as_tensor(random.sample(
                list(map(int, dataset[:, 0])), len(corrupted[1::2][:, 0][sample.nonzero()[:, 1]])), dtype=torch.int32)

        corrupted[1::2][:, 2][(sample==0).nonzero()[:, 1]] = \
            torch.as_tensor(random.sample(
                list(map(int, dataset[:, 2])), len(corrupted[1::2][:, 2][(sample==0).nonzero()[:, 1]])), dtype=torch.int32)

        # Guarantee that s!=o and that the sampled triple is not a true triple of any other dataset
        for i in range(len(corrupted[1::2])):
            while int(corrupted[1::2][i][0]) == int(corrupted[1::2][i][2]) \
                    and corrupted[1::2][i] in self.dataset.train \
                    and corrupted[1::2][i] in self.dataset.valid \
                    and corrupted[1::2][i] in self.dataset.test:
                if bool(random.getrandbits(1)) == True:
                    corrupted[1::2][i][2] = random.sample(list(dataset[:, 2]), 1)[0]
                else:
                    corrupted[1::2][i][0] = random.sample(list(dataset[:, 0]), 1)[0]

        # TODO: Create a function in util.sampler for that task. Then: Allow to choose from which entities to sample
        #  (e.g. from test, train and valid entities instead of only valid.

        # Save the labels per relation, since this will be needed frequently later
        p = corrupted[:, 1]
        rel_labels = {int(r): [labels[int((p == r).nonzero()[i])]
                               for i in range(len((p == r).nonzero()))] for r in p.unique()}

        return corrupted, labels, rel_labels

    def findThresholds(self, p, valid_scores, rel_scores, valid_labels, valid_data):
        """Method 1: Threshold is always one of the scores"""
        # Initialize accuracies, thresholds (and predictions)
        rel_accuracies = {int(r): -1 for r in p.unique()}
        rel_thresholds = {int(r): 0 for r in p.unique()}

        valid_scores = torch.as_tensor([float(valid_scores[i]) for i in range(len(valid_scores))])


        for r in p.unique():
            #Predict
            current_rel = (valid_data[:, 1] == r)
            true_labels = valid_labels[current_rel.nonzero()].type(torch.int)
            preds = (valid_scores[current_rel.nonzero()] >= rel_scores[int(r)]).type(torch.int)
            accuracy = [int(((true_labels==preds[i]).sum(dim=0)))/len(true_labels) for i in range(len(rel_scores[int(r)]))]

            rel_accuracies[int(r)] = max(accuracy)
            # Todo: Sometimes different scores can be the largest. Add condition, that always the largest/smalles/something else score that gives the maximum accuracy is chosen
            rel_thresholds[int(r)] = rel_scores[int(r)][accuracy.index(max(accuracy))]


        """Method 2: Search for best threshold in an interval
        https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py or https://github.com/dddoss/tensorflow-socher-ntn/blob/master/code/ntn_eval.py
        # Initialize accuracies, thresholds (and predictions)
        min_score = valid_scores.min()
        max_score = valid_scores.max()

        rel_accuracies = {int(r): -1 for r in p.unique()}
        rel_thresholds = {int(r): min_score for r in p.unique()}

        score = min_score

        # ORiginal implementation uses an interval 0.01, implemented for NTN model. In general the interval imo should depend on the range of the score values of the model
        # Suggestion: float((max_score-min_score)/len(valid_scores))
        interval = float((max_score-min_score)/len(valid_scores))
        valid_scores = torch.as_tensor([float(valid_scores[i]) for i in range(len(valid_scores))])

        while(score<=max_score):
            for r in p.unique():
                #Predict
                current_rel = (valid_data[:, 1] == r)
                true_labels = valid_labels[current_rel.nonzero()].type(torch.int)
                preds = (valid_scores[current_rel.nonzero()] >= score).type(torch.int)
                accuracy = int(((true_labels==preds).sum(dim=0)))/len(true_labels)

                if accuracy > rel_accuracies[int(r)]:
                    rel_accuracies[int(r)] = accuracy
                    rel_thresholds[int(r)] = score.clone()

            score += interval
        """
        return rel_thresholds, rel_accuracies

    def predict(self, rel_thresholds, test_scores, rel_scores, p_valid, p_test):

        rel_predictions = {int(r): torch.as_tensor([0]*len(rel_scores[int(r)])) for r in p_test.unique()}

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

        # Todo: Calculate accuracy and precision instead of using sklearn function
        metrics["Accuracy"] = float(accuracy_score(labels_in_test_list, pred_list))
        metrics["Precision"] = float(precision_score(labels_in_test_list, pred_list))

        precision_per_r = {}
        accuracy_per_r = {}
        for r in p_test.unique():
                precision_per_r[str(self.dataset.relations[int(r)])] = float(precision_score(rel_test_labels[int(r)], rel_predictions[int(r)]))
                accuracy_per_r[str(self.dataset.relations[int(r)])] = float(accuracy_score(rel_test_labels[int(r)], rel_predictions[int(r)]))

        metrics["Accuracy_per_Relation"] = accuracy_per_r

        metrics["Precision_Per_Relation"] = precision_per_r


        metrics["Untested relations due to missing in evaluation data"] = len(not_in_eval)

        return metrics

    # TODO-Question: We optimized the thresholds only for one randomly corrupted sample of the data.
    #  Another sample would give (a little) different results. How can we ǵet reproduceable results?

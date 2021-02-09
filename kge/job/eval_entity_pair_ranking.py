from kge.job import EvaluationJob, Job


class EntityPairRankingJob(EvaluationJob):
    """ Entity-pair ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        if self.__class__ == EntityPairRankingJob:
            for f in Job.job_created_hooks:
                f(self)

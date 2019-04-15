from kge.job import Job, Trace
from kge import Config
import itertools


class GridJob(Job):
    """Job to perform grid search.

    This job creates one subjob (a training job stored in a subfolder) for each
    point in the grid. The training jobs are then run in sequence and results
    analyzed.

    Interrupted grid searches can be resumed. Subjobs can also be resumed/run
    directly.

    """

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

    def resume(self):
        # no need to do anything here; run code automatically resumes
        pass

    def run(self):
        # read grid search optons range
        all_keys = []
        all_values = []
        all_indexes = []
        for k, v in Config.flatten(self.config.get('grid.options')).items():
            all_keys.append(k)
            all_values.append(v)
            all_indexes.append(range(len(v)))

        # create runs
        runs = []
        for indexes in itertools.product(*all_indexes):
            # obtain values for changed parameters
            values = list(map(lambda ik: all_values[ik[0]][ik[1]],
                              enumerate(list(indexes))))
            name = '_'.join(map(str, values))

            # create configuration of training job
            config = self.config.clone(name)
            config.set('job.type', 'train')
            for i, key in enumerate(all_keys):
                config.set(key, values[i])

            # save information about this run
            runs.append({'name': name,
                         'indexes': indexes,
                         'values': values,
                         'config': config})

        # create folders for runs (existing folders remain unmodified)
        for run in runs:
            run['config'].init_folder()

        # stop here?

        # now start running/resuming
        # TODO use a scheduler to run multiple jobs simultaneously?
        if self.config.get('grid.run'):
            for i, run in enumerate(runs):
                self.config.log("Starting training job {} ({}/{})..."
                                .format(run['name'], i+1, len(runs)))
                job = Job.create(run['config'], self.dataset, parent_job=self)
                job.resume()
                job.run()
        else:
            self.config.log(
                "Skipping running of training jobs as requested by user...")

        # read each runs trace file and produce a summary
        self.config.log("Reading results...")
        summary = []
        best = None
        best_metric = None
        metric_name = self.config.get('valid.metric')
        for i, run in enumerate(runs):
            config = run['config']
            trace = Trace(config.tracefile(), 'epoch')
            data = trace.filter(
                {'job': 'eval', 'scope': 'epoch', 'data': 'valid'})
            for row in data:
                for i, key in enumerate(all_keys):
                    row[key] = run['values'][i]
                row['folder'] = config.folder()
                metric = Trace.get_metric(row, metric_name)
                if not best or best_metric < metric:
                    best = row
                    best_metric = metric
                self.config.trace(**row)
            summary.append(data)
        self.config.log("And the winner is ({}={:.3f})..."
                        .format(metric_name, best_metric))
        best['valid_job_id'] = best['job_id']
        best['train_job_id'] = best['parent_job_id']
        del best['job'], best['job_id'], best['type'], best['parent_job_id']
        self.trace(echo=True, echo_prefix='  ', log=True,
                   metric_name=metric_name, metric_value=best_metric,
                   **best)

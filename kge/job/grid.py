from kge.job import Job
from kge.job import Trace
import itertools


class GridJob(Job):
    """Job to perform grid search.

    This job creates one subjob (a training job stored in a subfolder) for each
    point in the grid. The training jobs are then run in sequence and results
    analyzed.

    Interrupted grid searches can be resumed. Subjobs can also be resumed/run
    directly.

    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def resume(self):
        # no need to do anything here; run code automatically resumes
        pass

    def run(self):
        # read grid search optons range
        all_keys = []
        all_values = []
        all_indexes = []
        for kv in self.config.get('grid.options'):
            k, v = kv[0], kv[1]
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
                job = Job.create(run['config'], self.dataset)
                job.resume()
                job.run()
        else:
            self.config.log(
                "Skipping running of training jobs as requested by user...")

        # read each runs trace file and produce a summary
        self.config.log("Reading results...")
        summary = []
        best = None
        metric = self.config.get('valid.metric')
        for i, run in enumerate(runs):
            config = run['config']
            trace = Trace(config.tracefile(), 'epoch')
            data = trace.filter(
                { 'type': 'eval_er', 'scope': 'epoch', 'data': 'valid'})
            for row in data:
                for i, key in enumerate(all_keys):
                    row[key] = run['values'][i]
                    if not best or best[metric] < row[metric]:
                        best = row
                    self.config.trace(**row)
            summary.append(data)
        self.config.log("And the winner is...")
        best['type'] = 'grid'
        self.config.trace(echo=True, echo_prefix='  ', log=True, **best)

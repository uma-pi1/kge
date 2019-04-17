from kge.job import Job, Trace
from kge import Config


class SearchJob(Job):
    """Job to perform hyperparameter search.

    This job creates one subjob (a training job stored in a subfolder) for each
    hyperparameter setting. The training jobs are then run in sequence and
    results analyzed.

    Interrupted searches can be resumed. Subjobs can also be resumed/run
    directly. Configurations can be added/removed by modifying the config file.

    """
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

    def resume(self):
        # no need to do anything here; run code automatically resumes
        pass

    def run(self):
        # read search configurations and expand them to full configs
        search_configs = self.config.get('search.configurations')
        all_keys = set()
        for i in range(len(search_configs)):
            search_config = search_configs[i]
            folder = search_config['folder']
            del search_config['folder']
            config = self.config.clone(folder)
            config.set('job.type', 'train')
            flattened_search_config = Config.flatten(search_config)
            config.set_all(flattened_search_config)
            all_keys.update(flattened_search_config.keys())
            search_configs[i] = config

        # create folders for search_configs (existing folders remain
        # unmodified)
        for config in search_configs:
            config.init_folder()

        # now start running/resuming
        # TODO use a scheduler to run multiple jobs simultaneously?
        if self.config.get('search.run'):
            for i, config in enumerate(search_configs):
                self.config.log(
                    "Starting training job {} ({}/{})..."
                    .format(config.folder(), i+1, len(search_configs)))
                job = Job.create(config, self.dataset, parent_job=self)
                last_checkpoint = job.config.last_checkpoint()
                if last_checkpoint is None \
                   or last_checkpoint < job.config.get('train.max_epochs'):
                    def copy_to_search_trace(job, trace_entry):
                        for key in all_keys:
                            trace_entry[key] = config.get(key)
                            trace_entry['folder'] = config.folder()
                            self.config.trace(**trace_entry)
                    job.resume()
                    job.after_valid_hooks.append(copy_to_search_trace)
                    job.run()
                else:
                    self.config.log('Maximum number of epochs reached.')
        else:
            self.config.log(
                "Skipping running of training jobs as requested by user...")

        # read each search_configs trace file and produce a summary
        self.config.log("Reading results...")
        best = None
        best_metric = None
        metric_name = self.config.get('valid.metric')
        for config in search_configs:
            trace = Trace(config.tracefile(), 'epoch')
            data = trace.filter(
                {'job': 'eval', 'scope': 'epoch', 'data': 'valid'})
            for row in data:
                for key in all_keys:
                    row[key] = config.get(key)
                row['folder'] = config.folder()
                metric = Trace.get_metric(row, metric_name)
                if not best or best_metric < metric:
                    best = row
                    best_metric = metric
                # self.config.trace(**row)   # already done in after_valid_hook
        self.config.log("And the winner is ({}={:.3f})..."
                        .format(metric_name, best_metric))
        best['valid_job_id'] = best['job_id']
        best['train_job_id'] = best['parent_job_id']
        del best['job'], best['job_id'], best['type'], best['parent_job_id']
        self.trace(echo=True, echo_prefix='  ', log=True,
                   metric_name=metric_name, metric_value=best_metric,
                   **best)

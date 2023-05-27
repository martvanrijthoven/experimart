import wandb

from experimart.monitoring.tracking import Tracker


class WandbTracker(Tracker):
    def __init__(self, log_path, project):
        super().__init__(log_path=log_path)
        wandb.init(project=project, dir=log_path)

    def update(self, epoch_stats: dict):
        super().update(epoch_stats)
        wandb.log(epoch_stats)

    def save_parameters(self, parameters: dict):
        super().save_parameters(parameters)
        wandb.config.update(parameters)

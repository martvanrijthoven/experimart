import wandb

from experimart.monitoring.tracking import Tracker


class WandbTracker(Tracker):
    def __init__(self, log_path, project):
        super().__init__(log_path=log_path)
        wandb.init(project=project, dir=log_path)

    def update(self, statistics: dict):
        super().update(statistics)
        wandb.log(statistics)

    def save_parameters(self, parameters: dict):
        super().save_parameters(parameters)
        wandb.config.update(parameters)

import wandb
from experimart.monitoring.tracking import Tracker


class WandbTracker(Tracker):
    def __init__(self, log_path, project):
        super().__init__(log_path=log_path)
        wandb.init(project=project, dir=log_path)

    def update(self, epoch_stats: dict):
        super().update(epoch_stats)
        wandb.log(epoch_stats)

    def save(self, name: str, data: dict):  
        output_path = self._log_path / (name + '.yml')
        super().save(output_path=output_path, data=data)
        wandb.save(str(output_path))

    def update_config(self, config):
        wandb.config.update(config)

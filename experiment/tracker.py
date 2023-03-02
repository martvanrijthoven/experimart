from pathlib import Path
from typing import Union

import wandb
import yaml

from experiment.epoch import EpochStats


class Tracker:
    def __init__(self, logger, log_path):
        self._logger = logger 
        self._log_path = Path(log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

    def update(self, epoch_stats: EpochStats):
        self._logger.info(str(epoch_stats))

    def save(self, output_path: Union[str, Path], data: dict):
        with open(output_path, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

class WandbTracker(Tracker):
    def __init__(self, logger, project: str, log_path: Union[str, Path]):
        super().__init__(logger=logger, log_path=log_path)
        wandb.init(project=project, dir=log_path)

    def update(self, epoch_stats: EpochStats):
        super().update(epoch_stats)
        wandb.log(epoch_stats.dict())

    def save(self, name: str, data: dict):  
        output_path = self._log_path / (name + '.yml')

        super().save(output_path=output_path, data=data)
        wandb.save(str(output_path))

    def update_config(self, config):
        wandb.config.update(config)

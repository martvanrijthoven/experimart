import logging
from pathlib import Path
from typing import Union

import yaml


class Tracker:
    def __init__(self, log_path):
        self._log_path = Path(log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    @property
    def log_path(self):
        return self._log_path

    def update(self, epoch_stats: dict):
        self._logger.info(str(epoch_stats))

    def save(self, output_path: Union[str, Path], data: dict):
        with open(output_path, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

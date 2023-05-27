import logging
from pathlib import Path
from pprint import pprint

import yaml


class Tracker:
    def __init__(self, log_path):
        self._log_path = Path(log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    @property
    def log_path(self):
        return self._log_path

    def update(self, statistics: dict):
        self._logger.info(str(statistics))

    def save_parameters(self, parameters: dict):
        with open(self._log_path / "parameters.yml", "w") as outfile:
            yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)
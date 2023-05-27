import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s", "%I:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)


logger.addHandler(stream_handler)


class LocalTracker:
    def __init__(self, log_path):
        self._log_path = Path(log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self):
        return self._log_path

    def update(self, statistics: dict):
        logger.info(str(statistics))

    def save_parameters(self, parameters: dict):
        with open(self._log_path / "parameters.yml", "w") as outfile:
            yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

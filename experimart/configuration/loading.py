from copy import deepcopy
from pathlib import Path

from dicfg import ConfigReader


NAME = "experimart"
MAIN_CONFIG_PATH = Path(__file__).parent.parent / "configuration" / f"{NAME}.yml"


def load_config(config, search_paths, presets):
    config_reader = ConfigReader(
        name=NAME, main_config_path=MAIN_CONFIG_PATH, search_paths=search_paths
    )
    config = config_reader.read(config, presets=presets)["default"]
    return deepcopy(config)

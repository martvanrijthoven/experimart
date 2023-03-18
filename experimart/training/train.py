from pathlib import Path

import click
from dicfg import ConfigReader, build_config
from tqdm import tqdm
from pprint import pprint

NAME = "experimart"
MAIN_CONFIG_PATH = Path(__file__).parent.parent / "configuration" / f"{NAME}.yml"


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--config", "-c", type=Path, required=False, default=None)
@click.option("--search_paths", "-s", multiple=True, type=Path, required=False)
@click.option("--presets", "-p", multiple=True, required=False)
def run(config=None, search_paths=(), presets=()):
    reader = ConfigReader(name=NAME, main_config_path=MAIN_CONFIG_PATH, search_paths=search_paths)
    config = reader.read(config, presets=presets)["default"]
    pprint(config, sort_dicts=False)
    training = build_config(config)
    epoch_iterator = training["epoch_iterator"]
    for _ in range(len(epoch_iterator)):
        next(epoch_iterator)

if __name__ == "__main__":
    run()

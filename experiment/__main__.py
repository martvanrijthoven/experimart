from pathlib import Path

import click
from tqdm import tqdm
from dicfg import ConfigReader, build_config
from experiment.tracker import Tracker

NAME = "experiment"
MAIN_CONFIG_PATH = Path(__file__).parent / "configuration" / f"{NAME}.yml"


@click.command()
@click.option("--experiment_config", type=Path, required=False, default=None)
@click.option("--presets", "-p", multiple=True)
def train(experiment_config = None, presets = ()):
    reader = ConfigReader(name=NAME, main_config_path=MAIN_CONFIG_PATH)
    config = reader.read(experiment_config,  presets=presets)
    # print(config)
    experiment = build_config(config["default"])
    # print(experiment['training_iterator'].dataset)
    tracker: Tracker = experiment["tracker"]
    tracker.save(name=NAME, data=config)

    epoch_iterator = experiment["epoch_iterator"]

    for epoch_stats in tqdm(next(epoch_iterator), total=len(epoch_iterator)):
        tracker.update(epoch_stats)

train()
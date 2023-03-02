from pathlib import Path

import click
from tqdm import tqdm
from dicfg import ConfigReader, build_config
from experiment.tracker import Tracker
from pprint import pprint

NAME = "experiment"
MAIN_CONFIG_PATH = Path(__file__).parent / "configuration" / f"{NAME}.yml"


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--experiment_config", type=Path, required=False, default=None)
@click.option("--presets", "-p", multiple=True)
def train(experiment_config = None, presets = ()):
    try:
        reader = ConfigReader(name=NAME, main_config_path=MAIN_CONFIG_PATH)
        config = reader.read(experiment_config, presets=presets)
        pprint(config, sort_dicts=False)
        experiment = build_config(config["default"], log_folder=Path(config['default']['log_path']))
    
        tracker: Tracker = experiment["tracker"]
        tracker.save(name=NAME, data=config)

        epoch_iterator = experiment["epoch_iterator"]
        # for epoch_stats in tqdm(next(epoch_iterator), total=len(epoch_iterator)):
        #     tracker.update(epoch_stats)

    finally:
        try:
            experiment['training_iterator'].stop()
            experiment['validation_iterator'].stop()
        except:
            pass


train()
from pathlib import Path

import click
from dicfg import build_config

from experimart.configuration.loading import load_config
from experimart.training.epoch import EpochIterator


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

    configurations: dict = load_config(config, search_paths, presets)
    objects: dict = build_config(configurations)

    for tracker in objects["trackers"]:
        tracker.save_parameters(configurations)

    epoch_iterator: EpochIterator = objects["epoch_iterator"]
    for epoch in epoch_iterator:
        epoch()


if __name__ == "__main__":
    run()

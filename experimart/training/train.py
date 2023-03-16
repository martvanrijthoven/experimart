from pathlib import Path

import click
from dicfg import ConfigReader, build_config
from tqdm import tqdm


NAME = "training"
MAIN_CONFIG_PATH = Path(__file__).parent.parent / "configuration" / f"{NAME}.yml"


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--config", "-c", type=Path, required=False, default=None)
@click.option("--presets", "-p", multiple=True)
def run(config=None, presets=()):

    reader = ConfigReader(name=NAME, main_config_path=MAIN_CONFIG_PATH)
    config = reader.read(config, presets=presets)["default"]
    training = build_config(config, log_folder=Path(config["log_path"]))
    epoch_iterator = training["epoch_iterator"]
    for _ in range(len(epoch_iterator)):
        next(epoch_iterator)

if __name__ == "__main__":
    run()

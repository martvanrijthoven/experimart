import wandb

class WandbTracker:
    def __init__(self, log_path, project):
        wandb.init(project=project, dir=log_path)

    def update(self, statistics: dict):
        wandb.log(statistics)

    def save_parameters(self, parameters: dict):
        wandb.config.update(parameters)

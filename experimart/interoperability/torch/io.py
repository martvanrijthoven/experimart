import os
from pathlib import Path
from typing import Union

import torch

def to_cuda(model):
    return model.cuda()

def get_model_parameters(model):
    return model.parameters()


class TorchModelIO:
    def __init__(self, model, output_folder: Union[str, Path], suffix: str):
        self._model = model
        self._output_path = Path(output_folder) / f"{suffix}.pth"

        if not os.path.exists(output_folder):
            raise ValueError(f"Output folder '{output_folder}' does not exist.")
        if not os.access(output_folder, os.W_OK):
            raise ValueError(f"Output folder '{output_folder}' is not writable.")

    def __call__(self, epoch: int, train_output, validation_output):
        torch.save(self._model.state_dict(), self._output_path)

    @classmethod
    def load(cls, model, saved_model_path: Union[str, Path]):
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)
        return model

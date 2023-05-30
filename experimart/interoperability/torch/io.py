import os
from pathlib import Path
from typing import Union

import torch

def to_cuda(model):
    return model.cuda()

def get_model_parameters(model):
    return model.parameters()

def convert_data_to_device(data, label, device):
    data = torch.tensor(data, device=device).float() / 255.0
    label = torch.tensor(label, device=device).long()
    return data, label

class TorchModelTracker:
    def __init__(self, model, output_folder: Union[str, Path], suffix: str):
        self._model = model
        self._output_path = Path(output_folder) / f"{suffix}.pth"

        if not os.path.exists(output_folder):
            raise ValueError(f"Output folder '{output_folder}' does not exist.")
        if not os.access(output_folder, os.W_OK):
            raise ValueError(f"Output folder '{output_folder}' is not writable.")

    def update(self, statistics: dict):
        torch.save(self._model.state_dict(), self._output_path)

    def save_parameters(self, parameters):
        pass

    @classmethod
    def load(cls, model, saved_model_path: Union[str, Path]):
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)
        return model

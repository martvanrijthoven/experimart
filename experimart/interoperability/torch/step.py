from typing import Callable, Iterable, List, Tuple

import torch
from abc import ABC
from experimart.training.step import StepIterator


def get_cuda_data(data, label):
    data = torch.tensor(data, device="cuda").float()
    label = torch.tensor(label, device="cuda").long()
    return data, label


class TorchStepComponents:
    def __init__(self, optimizer, criterion, scheduler):
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def criterion(self):
        return self._criterion

    @property
    def scheduler(self):
        return self._scheduler


class TorchStepIterator(ABC, StepIterator):
    def __init__(
        self,
        model: torch.nn.Module,
        data_iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        num_steps: int,
        components: TorchStepComponents,
        metrics: List[Callable] = None,
    ):
        super().__init__(model, data_iterator, num_steps, metrics)
        self._components = components


class TorchTrainingStepIterator(TorchStepIterator):
    def steps(self):
        self._model.train()
        for _ in range(len(self)):
            data, label, *_ = next(self._data_iterator)
            data, label = get_cuda_data(data, label)
            self._components.optimizer.zero_grad()
            output = self._model(data)["out"]
            loss = self._components.criterion(output, label)
            loss.backward()
            self._components.optimizer.step()
            metrics = self.get_metrics(label, output)
            metrics['leanning_rate'] = self._components.scheduler.get_lr()
            yield {"loss": loss.item(), **metrics}
        self._components.scheduler.step()


class TorchValidationStepIterator(TorchStepIterator):
    def steps(self):
        self._model.eval()
        with torch.no_grad():
            for _ in range(len(self)):
                data, label, *_ = next(self._data_iterator)
                data, label = get_cuda_data(data, label)
                output = self._model(data)
                output = output["out"]
                loss = self._components.criterion(output, label)
                metrics = self.get_metrics(label, output)
                yield {"loss": loss.item(), **metrics}

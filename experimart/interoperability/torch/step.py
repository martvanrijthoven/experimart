from typing import Callable, Iterable, List, Tuple
from experimart.interoperability.torch.io import convert_data_to_device

import torch
from abc import ABC
from experimart.training.step import StepIterator


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
        device: str = 'cuda',
        multi_inputs: bool = False,
    ):
        super().__init__(model, data_iterator, num_steps, metrics)
        self._components = components
        self._device = device
        self._multi_inputs = multi_inputs

    def _get_data(self):
        data, label = next(self._data_iterator)
        return convert_data_to_device(data, label, device=self._device)

    def _get_output(self, data):
        if self._multi_inputs:
            return self._model(*data)["out"]
        return self._model(data)["out"]

    def _get_loss(self, output, label):
        return self._components.criterion(output, label)


class TorchTrainingStepIterator(TorchStepIterator):
    def _update_training_metrics(self):
        return {"learning_rate": self._components.scheduler.get_lr()}

    def steps(self):
        self._model.train()
        for _ in range(len(self)):
            data, label = self._get_data()
            data, label = convert_data_to_device(data, label)
            self._components.optimizer.zero_grad()
            output = self._get_output(data)
            loss = self._get_loss(output, label)
            loss.backward()
            self._components.optimizer.step()
            metrics = self._get_metrics(label, output)
            metrics.update(self._update_training_metrics())
            yield {"loss": loss.item(), **metrics}
        self._components.scheduler.step()


class TorchValidationStepIterator(TorchStepIterator):
    def _update_validation_metrics(self):
        return {}

    def steps(self):
        self._model.eval()
        with torch.no_grad():
            for _ in range(len(self)):
                data, label = self._get_data()
                data, label = convert_data_to_device(data, label)
                output = self._get_output(data)
                loss = self._get_loss(output, label)
                metrics = self._get_metrics(label, output)
                metrics.update(self._update_validation_metrics())
                yield {"loss": loss.item(), **metrics}

from typing import Callable, Iterable, List, Tuple

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
    ):
        super().__init__(model, data_iterator, num_steps, metrics)
        self._components = components

    def _get_data(self):
        data, label = next(self._data_iterator)
        data = torch.tensor(data, device="cuda").float() // 225.0
        label = torch.tensor(label, device="cuda").long()
        return data, label

    def _get_output(self, data):
        return self._model(data)["out"]

    def _get_loss(self, output, label):
        return self._components.criterion(output, label)


class TorchTrainingStepIterator(TorchStepIterator):
    def _update_training_metrics(self):
        return {"leanning_rate": self._components.scheduler.get_lr()}

    def steps(self):
        self._model.train()
        for _ in range(len(self)):
            data, label = self._get_data()
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
                output = self._get_output(data)
                loss = self._get_loss(output, label)
                metrics = self._get_metrics(label, output)
                metrics.update(self._update_validation_metrics)
                yield {"loss": loss.item(), **metrics}

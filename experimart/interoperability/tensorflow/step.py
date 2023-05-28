from abc import ABC
from typing import Iterable

from experimart.interoperability.tensorflow.components import (
    TensorLearningRateScheduler,
)
from experimart.training.step import StepIterator


class TensorFlowStepComponents:
    def __init__(self, lr_scheduler: TensorLearningRateScheduler):
        self._lr_scheduler = lr_scheduler

    @property
    def lr_scheduler(self):
        return self._lr_scheduler


class TensorflowStepIterator(ABC, StepIterator):
    def __init__(
        self,
        model,
        data_iterator: Iterable,
        num_steps: int,
        components: TensorFlowStepComponents = None,
        metrics=None,
    ):
        super().__init__(model, data_iterator, num_steps, metrics)
        self._components = components

    def _update_learning_rate(self):
        if self._components is not None and self._components.lr_scheduler is not None:
            return self._components.lr_scheduler()
        return None

class TensorflowTrainingStepIterator(TensorflowStepIterator):
    def steps(self):
        for _ in range(len(self)):
            x_batch, y_batch, _ = next(self._data_iterator)
            yield self._model.train_on_batch(
                x=x_batch, y=y_batch, return_dict=True, metrics=self._metrics
            )


class TensorflowValidationStepIterator(TensorflowStepIterator):
    def steps(self):
        lr = self._update_learning_rate()
        for _ in range(len(self)):
            metrics = {} if lr is None else {"learning_rate": lr}
            x_batch, y_batch, _ = next(self._data_iterator)
            predictions = self._model.predict_on_batch(x=x_batch, argmax=False)
            metrics.update(self.get_metrics(y_batch, predictions))
            yield metrics

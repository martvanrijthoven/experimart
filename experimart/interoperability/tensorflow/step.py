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

    def _get_data(self):
        return next(self._data_iterator)


class TensorflowTrainingStepIterator(TensorflowStepIterator):
    def _update_training_metrics(self):
        return {}

    def steps(self):
        for _ in range(len(self)):
            x, y = self._get_data()
            metrics = self._model.train_on_batch(
                x=x, y=y, return_dict=True, metrics=self._metrics
            )
            metrics.update(self._update_training_metrics())


class TensorflowValidationStepIterator(TensorflowStepIterator):
    def _update_validation_metrics(self):
        return {}

    def steps(self):
        self._update_learning_rate()
        for _ in range(len(self)):
            x, y = self._get_data()
            predictions = self._model.predict_on_batch(x=x, argmax=False)
            metrics = self.get_metrics(y, predictions)
            metrics.update(self._update_validation_metrics())
            yield metrics

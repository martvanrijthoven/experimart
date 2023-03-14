from typing import Iterable
from training.step import StepIterator


class TensorLearningRateScheduler:
    def __init__(
        self,
        model,
        decay_rate=0.5,
        decay_steps=[2, 4, 1000, 5000, 10_000, 50_000, 100_000],
    ):
        self._model = model
        self._lr = self._model.optimizer.lr.numpy()
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self._index = 0

    def __call__(self):
        if self._index in self._decay_steps:
            self._lr *= self._decay_rate
            self._model.optimizer.lr.assign(self._lr)
        self._index += 1
        return self._lr


class TensorFlowStepComponents:
    def __init__(self, lr_scheduler: TensorLearningRateScheduler):
        self._lr_scheduler = lr_scheduler

    @property
    def lr_scheduler(self):
        return self._lr_scheduler


class TensorflowStepIterator(StepIterator):
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
            metrics = {} if lr is None else {'learning_rate': lr}    
            x_batch, y_batch, _ = next(self._data_iterator)
            predictions = self._model.predict_on_batch(x=x_batch, argmax=False)
            metrics.update(self.get_metrics(y_batch, predictions))
            yield metrics

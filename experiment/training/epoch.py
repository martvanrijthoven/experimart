from collections.abc import Iterator
from typing import Callable, Protocol

from pandas import DataFrame

from experiment.training.step import StepIterator


class EpochCallback(Protocol):
    def __call__(epoch: int, train_output: DataFrame, validation_output: DataFrame):
        raise NotImplementedError()


class EpochIterator(Iterator):
    def __init__(
        self,
        num_epochs: int,
        training_step_iterator: StepIterator,
        validation_step_iterator: StepIterator,
        epoch_callbacks: Callable,
    ):
        self._num_epochs = num_epochs
        self._training_step_iterator = training_step_iterator
        self._validation_step_iterator = validation_step_iterator
        self._epoch_callbacks = epoch_callbacks
        self._epoch = 0

    def __len__(self):
        return self._num_epochs

    def __next__(self):
        if self._epoch == len(self):
            raise StopIteration()
        
        train_output = DataFrame([out for out in next(self._training_step_iterator)])
        validation_output = DataFrame([out for out in next(self._validation_step_iterator)])

        for epoch_callback in self._epoch_callbacks:
            epoch_callback(self._epoch, train_output, validation_output)
        
        self._epoch += 1

    




from collections.abc import Iterator
from typing import Callable, Protocol, Tuple

from pandas import DataFrame

from experimart.training.step import StepIterator


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

    def __len__(self) -> int:
        return self._num_epochs

    def __call__(self):
        self._epoch += 1
        
        training = DataFrame([out for out in next(self._training_step_iterator)])
        validation = DataFrame([out for out in next(self._validation_step_iterator)])
        outputs={'training': training, 'validation': validation}

        for epoch_callback in self._epoch_callbacks:
            epoch_callback(self._epoch, outputs=outputs)
        
    def __next__(self) -> Callable:
        if self._epoch >= len(self):
            raise StopIteration()
        return self

from collections.abc import Iterator
from dataclasses import dataclass, asdict
from statistics import mean


@dataclass
class EpochStats:
    epoch: int
    mean_train_loss: float
    mean_validation_loss: float
    dict = asdict


class EpochIterator(Iterator):
    def __init__(
        self,
        num_steps,
        tracker,
        model,
        training_step_iterator,
        validation_step_iterator,
    ):

        self._num_steps = num_steps
        self._tracker = tracker
        self._model = model

        self._training_step_iterator = training_step_iterator
        self._validation_step_iterator = validation_step_iterator

    def __len__(self):
        return self._num_steps

    def __next__(self):
        return self.run()

    def run(self):
        for epoch in range(self._num_steps):
            train_loss = mean([loss for loss in self._training_step_iterator])
            validation_loss = mean([loss for loss in self._validation_step_iterator])
            yield EpochStats(epoch, train_loss, validation_loss)

from collections.abc import Iterator
from typing import Any, Dict


class StepIterator(Iterator):
    def __init__(self, model, data_iterator, num_steps, metrics=None):
        self._model = model
        self._data_iterator = data_iterator
        self._num_steps = num_steps
        self._index = 0
        self._metrics = metrics

    def __len__(self):
        return self._num_steps

    def __next__(self):
        return self.steps()

    def _get_metrics(self, y_true, y_pred):
        if self._metrics is None:
            return {}
        return {name: metric(y_true, y_pred) for name, metric in self._metrics.items()}

    def _get_data(self):
        return next(self._data_iterator)

    def steps(self) -> Dict[str, Any]:
        for _ in range(len(self)):
            yield {'loss': 0.5}

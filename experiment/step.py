from collections.abc import Iterator


class StepIterator(Iterator):
    def __init__(self, model, data_iterator, num_steps):
        self._model = model
        self._data_iterator = data_iterator
        self._num_steps = num_steps
        self._index = 0

    def __len__(self):
        return self._num_steps

    def __next__(self):
        return self.steps()

    def steps(self):
        for _ in range(len(self)):
            raise NotImplementedError

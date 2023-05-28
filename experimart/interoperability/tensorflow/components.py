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
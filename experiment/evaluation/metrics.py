from typing import Protocol


class ConfusionMatrixMetric(Protocol):
    def update(self):
        ...

    def get_matrix(self):
        ...

    def reset(self):
        ...
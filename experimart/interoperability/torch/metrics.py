from typing import List

import numpy as np
import torch
from pycm import ConfusionMatrix


class TorchConfusionMatrixMetricCollector:
    def __init__(self, metric: str):
        self._metric = metric

    def __call__(self, cms: List["TorchConfusionMatrixMetric"]):
        return getattr(cms.iloc[-1].get_matrix(), self._metric)


class TorchConfusionMatrixMetric:
    """
    Computes a confusion matrix for a multiclass segmentation task.
    """

    def __init__(self, num_classes, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self._device = device
        self.reset()

    def __call__(self, y_true, y_pred):
        y_pred = y_pred.argmax(1)
        
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        mask = y_true != 0

        y_true = y_true[mask] - 1
        y_pred = y_pred[mask] - 1

        # Compute the confusion matrix for the current batch using matrix operations
        y_true_onehot = torch.nn.functional.one_hot(
            y_true, num_classes=self.num_classes
        ).float()
        y_pred_onehot = torch.nn.functional.one_hot(
            y_pred, num_classes=self.num_classes
        ).float()

        # Update the confusion matrix for the current batch and accumulate it with the existing confusion matrix
        self.values += torch.matmul(y_true_onehot.t(), y_pred_onehot)
        return self

    def get_matrix(self):
        cm = ConfusionMatrix(matrix=self.values.cpu().numpy().astype(np.int64))
        self.reset()
        return cm

    def reset(self):
        self.values = torch.zeros(
            (self.num_classes, self.num_classes), device=self._device
        )

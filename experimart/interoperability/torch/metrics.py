import numpy as np
import torch


class TorchConfusionMatrixMetric(torch.nn.Module):
    """
    Computes a confusion matrix for a multiclass segmentation task.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cm = torch.zeros((num_classes, num_classes))

    def forward(self, y_true, y_pred):
        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)

        # Compute the confusion matrix for the current batch using matrix operations
        y_true_onehot = torch.nn.functional.one_hot(
            y_true, num_classes=self.num_classes
        ).float()
        y_pred_onehot = torch.nn.functional.one_hot(
            y_pred, num_classes=self.num_classes
        ).float()
        cm = torch.matmul(y_true_onehot.t(), y_pred_onehot)

        # Update the confusion matrix for the current batch and accumulate it with the existing confusion matrix
        self.cm += cm

    def get_matrix(self, reset=True):
        # Cast the final confusion matrix to a NumPy array
        cm = self.cm.numpy().astype(np.int64)
        if reset:
            self.reset()
        return cm

    def reset(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes))

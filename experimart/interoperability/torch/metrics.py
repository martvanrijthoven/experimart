import numpy as np
import torch


class TorchConfusionMatrixMetric(torch.nn.Module):
    """
    Computes a confusion matrix for a multiclass segmentation task.
    """

    def __init__(self, num_classes, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self._device = device
        self.reset()

    def forward(self, y_true, y_pred):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        mask = y_true != 0
        
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Compute the confusion matrix for the current batch using matrix operations
        y_true_onehot = torch.nn.functional.one_hot(
            y_true, num_classes=self.num_classes
        ).float()
        y_pred_onehot = torch.nn.functional.one_hot(
            y_pred, num_classes=self.num_classes
        ).float()
        
        # Update the confusion matrix for the current batch and accumulate it with the existing confusion matrix
        self.cm += torch.matmul(y_true_onehot.t(), y_pred_onehot)

    def get_matrix(self, reset=True):
        # Cast the final confusion matrix to a NumPy array
        cm = self.cm.cpu().numpy().astype(np.int64)
        if reset:
            self.reset()
        return cm

    def reset(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes), device=self._device)
        
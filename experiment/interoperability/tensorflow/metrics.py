from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
import numpy as np


class TensorflowConfusionMatrixMetric(Metric):
    """
    Computes a confusion matrix for a multiclass segmentation task.
    """
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.cm = self.add_weight(name='cm', shape=(num_classes, num_classes), initializer='zeros')

    def update(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)

        # Compute the confusion matrix for the current batch using tf.math.confusion_matrix() function
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        cm = K.cast(cm, dtype=K.floatx())

        # Update the confusion matrix for the current batch and accumulate it with the existing confusion matrix
        self.cm.assign_add(cm)

    def get_matrix(self, reset=True):
        # Cast the final confusion matrix to a NumPy array
        cm = K.eval(self.cm).astype(np.int64)
        if reset:
            self.reset()
        return cm

    def reset(self):
        K.batch_set_value([(v, K.zeros((self.num_classes, self.num_classes))) for v in self.variables])

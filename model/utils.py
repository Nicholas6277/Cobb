import torch
import torch.nn as nn
import numpy as np


class Dice(nn.Module):
    def __init__(self, eps=1e-7, threshold=0.5):
        super(Dice, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.activation = torch.sigmoid

    @property
    def __name__(self):
        return 'Dice'

    def _threshold(self, x):
        return (x > self.threshold).type(x.dtype)

    def forward(self, y_pr, y_gt):
        """
        Compute the Dice score (a measure of overlap between the prediction and the ground truth).

        Args:
            y_pr: Predicted tensor (logits or probabilities)
            y_gt: Ground truth tensor (binary mask)

        Returns:
            dice score: Scalar representing the Dice coefficient.
        """
        y_pr = self.activation(y_pr)
        y_pr = self._threshold(y_pr)

        tp = torch.sum(y_gt * y_pr)  # True positives
        fp = torch.sum(y_pr) - tp  # False positives
        fn = torch.sum(y_gt) - tp  # False negatives

        score = torch.div((2 * tp + self.eps), (2 * tp + fn + fp + self.eps))
        return score


class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 1:
            self.mean = 0.0 + self.sum
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


def to_string(logs):
    """
    Converts a dictionary of logs into a formatted string.

    Args:
        logs: Dictionary of log values (key-value pairs).

    Returns:
        str: A formatted string of the log values.
    """
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s

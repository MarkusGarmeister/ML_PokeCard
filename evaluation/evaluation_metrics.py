import numpy as np
from sklearn.metrics import accuracy_score


class EvaluationMetrics:

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

        self.accuracy = accuracy_score(self.y_true, self.y_pred)

    def print_eval_metrics(self):
        print(f"Accuracy: {self.accuracy}")

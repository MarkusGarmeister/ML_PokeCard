import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


class EvaluationMetrics:

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = self._to_class_indices(y_true)
        self.y_pred = self._to_class_indices(y_pred)

        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.f1_macro = f1_score(self.y_true, self.y_pred, average="macro")

    @staticmethod
    def _to_class_indices(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 2:
            return np.argmax(y, axis=1)
        return y

    def print_eval_metrics(self):
        print(f"Accuracy: {self.accuracy}")
        print(f"F1 Score (Macro): {self.f1_macro}")

    def print_classification_report(self, class_names):
        print(
            classification_report(
                self.y_true, self.y_pred, target_names=class_names, zero_division=0
            )
        )

    def plot_confusion_matrix(self, class_names, title="Confusion Matrix"):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        _, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

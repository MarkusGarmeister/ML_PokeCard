import numpy as np


class RandomGuessingBenchmark:
    def __init__(self):
        self.num_classes = 5

    def predict(self, X: np.ndarray) -> np.ndarray:

        n_samples = len(X)
        predictions = np.zeros((n_samples, self.num_classes))
        random_classes = np.random.randint(0, self.num_classes, size=n_samples)
        predictions[np.arange(n_samples), random_classes] = 1
        return predictions

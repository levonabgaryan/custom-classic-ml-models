import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k: int = 5):
        self.k = k
        if k % 2 == 0:
            raise ValueError("k must be odd")

        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = np.array(x)
        self.y_train = np.array(y)

    def _get_single_prediction(self, x_test_row) -> int:
        # Calculate all distances of x_test_row from all points
        distances = np.sqrt(np.sum((self.x_train - x_test_row) ** 2, axis=1))

        # Get first k indices
        k_indices = np.argsort(distances)[:self.k]

        # Get k nearest labels
        k_nearest_labels = self.y_train[k_indices]

        # Voting
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, x_test):
        if self.x_train is None:
            raise RuntimeError("Model must be fitted before predicting")

        x_test = np.array(x_test)
        return np.array([self._get_single_prediction(row) for row in x_test])

    def score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(predictions == np.array(y_test))
import numpy as np
import numpy.typing as npt
from typing import Optional


class LogisticRegressionCustom:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) -> None:
        self.lr = learning_rate
        self.iterations = iterations
        self.weights: Optional[npt.NDArray[np.float64]] = None

    @staticmethod
    def _sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = np.clip(z, -700, 700)  #
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _add_intercept(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    def fit(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        """
        Train the model using Gradient Descent.
        Args:
            x: Features matrix (n_samples, n_features)
            y: Target labels (n_samples, 1) or (n_samples,)
        """
        # Data preparation
        x_full = self._add_intercept(x)
        x_full = np.array(x_full)

        y = np.array(y).reshape(-1, 1)

        n_samples, n_features = x_full.shape
        # Init weights using zeroes
        self.weights = np.zeros((n_features, 1))

        # Gradient disent
        for _ in range(self.iterations):
            # 1. Linear combination
            z = x_full @ self.weights
            # 2. Sigmoid prediction
            predictions = self._sigmoid(z)
            # 3. Calculate the gradient
            # Formula: (1/m) * X^T * (predictions - y)
            gradient = (x_full.T @ (predictions - y)) / n_samples
            # 4. Refresh weights
            self.weights -= self.lr * gradient

        print(f"Model trained. weights: {self.weights.flatten()}")

    def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns probability estimates (0 to 1)."""
        if self.weights is None:
            raise RuntimeError("Model not fitted")
        x_full = self._add_intercept(x)
        return self._sigmoid(x_full @ self.weights)

    def predict(self, x: npt.NDArray[np.float64], threshold: float = 0.5) -> npt.NDArray[np.int64]:
        """Returns binary class labels (0 or 1)."""
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).astype(np.int64)

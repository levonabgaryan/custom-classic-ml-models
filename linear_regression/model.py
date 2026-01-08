import numpy as np
import numpy.typing as npt
from typing import Optional


class LinearRegression:
    def __init__(self) -> None:
        """Init model. Weights at start is empty"""
        self.betas: Optional[npt.NDArray[np.float64]] = None

    @staticmethod
    def _add_intercept(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Add 1 column for calculating bias.
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    def fit(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        """
        Train the model using the Ordinary Least Squares (OLS) method.

        This method uses a geometric approach (Normal Equation) to find the
        exact solution analytically without using gradient-based optimization.

        Note:
            If the dataset is very large (massive number of features), this method
            may fail or be extremely slow due to the O(n^3) complexity of
            matrix inversion/solving. In such cases, consider using a
            Gradient Descent based model.

        Args:
            x: Features matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples, 1).
        """
        x_full = self._add_intercept(x)

        # beta = (X^T * X)^-1 * X^T * y
        XTX = x_full.T @ x_full
        XTy = x_full.T @ y
        self.betas = np.linalg.solve(XTX, XTy)

        print(f"Model is fited. Count of weights: {len(self.betas)}")

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Calculate prediction y_pred = X * beta.
        if self.betas is None:
            raise RuntimeError("Model is not trained, call method .fit() at first")

        x_full = self._add_intercept(x)
        return x_full @ self.betas
import numpy as np
from get_prediction import get_prediction

def predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.array:
  """
  Predict the classes for a set of input points using k-nearest neighbors.

  Arguments:
    X_train (np.ndarray): The training data points.
    y_train (np.ndarray): The labels for the training data.
    X_test (np.ndarray): The input data points to consider.
    k (int): The number of nearest neighbors to consider.

  Returns:
    (np.ndarray): An array of predicted classes for the input points.
  """
  y_preds = np.zeros(len(X_test), dtype=np.int64)

  for idx, x_input in enumerate(X_test):
    y_preds[idx] = get_prediction(x_input, X_train, y_train, k=k)

  return y_preds

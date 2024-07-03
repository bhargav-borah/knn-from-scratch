import numpy as np
from mean import mean
from get_prediction import get_prediction
from distance import distance

def predict(X: np.ndarray, y: np.array, k: int) -> np.ndarray:
  """
  Predict the target values for a set of input points using the k-Nearest Neighbors algorithm.

  Arguments:
    X (np.ndarray): The input points for which predictions are to be made.
    y (np.ndarray): The target values corresponding to the training data points.
    k (int): The number of nearest neighbors to be considered for making the predictions.

  Returns:
    (np.ndarray): The predicted target values for the input points.
  """
  y_preds = np.zeros(len(X))

  for idx, x_input in enumerate(X):
    y_preds[idx] = get_prediction(x_input, X, y, k)

  return y_preds

import numpy as np
from mean import mean
from distance import distance

def get_prediction(x_input: np.ndarray, X: np.ndarray, y: np.ndarray, k: int) -> float:
  """
  Predict the target value for a given point x_input using the k-Nearest Neighbors algorithm.

  Arguments:
    x_input (np.ndarray): The point for which a target value has to be predicted.
    X (np.ndarray): The training data points.
    y (np.ndarray): The target values corresponding to the training data points.
    k (int): The number of nearest neighbors to consider for the prediction.

  Returns:
    (float): The target value predicted for the given input point.
  """
  assert k > 0 and isinstance(k, int), "'k' must be a positive integer."
  distance_target = {}

  for idx, x in enumerate(X):
    if not np.array_equal(x, x_input):
      distance_target[distance(x_input, x)] = y[idx]

  sorted_distance_target = sorted(distance_target.items())[:k]

  return mean(sorted_distance_target)

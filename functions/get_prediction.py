import numpy as np
from collections import Counter

def get_prediction(x_input: np.array, X: np.ndarray, y: np.array, k: int) -> int:
  """
  Get the prediction for a single input using k-nearest neighbors.

  Arguments:
    x_input (np.array): The point to classify.
    X (np.ndarray): The training data points.
    y (np.array): The labels for the training data.
    k (int): The number of nearest neighbors to consider.

  Returns:
    (int): The predicted class for the input point.
  """
  assert k > 0 and isinstance(k, int), "k must be a positive integer."

  distance_index_dict = {}

  for idx, x in enumerate(X):
    if not np.array_equal(x, x_input):
      distance_index_dict[distance(x, x_input, distance_type='euclidean')] = y[idx]

  k_nearest_points = sorted(distance_index_dict.items())[:k]

  prediction = Counter([class_ for _, class_ in k_nearest_points]).most_common(1)[0][0]

  return prediction

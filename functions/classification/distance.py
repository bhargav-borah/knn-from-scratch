import numpy as np

def distance(v1: np.array, v2: np.array, distance_type : str = 'euclidean', p : int = None) -> float:
  """
  Compute the distance between two points (vectors).

  Arguments:
    v1 (np.ndarray): First vector.
    v2 (np.ndarray): Second vector
    distance_type (str): Type of default metric. Default is 'euclidean'.
    p (int): Parameter for Minkowski distance. Default is None.

  Returns:
    (float): The distance betweem two vectors.
  """
  if distance_type == 'euclidean':
    return np.sqrt(np.sum((v1 - v2) ** 2))

  elif distance_type == 'manhattan':
    return np.sum(np.abs(v1 - v2))

  elif distance_type == 'minkowski' and p is not None:
    return np.sum((v1 - v2) ** p) ** (1 / p)

  elif distance_type == 'chebyshev':
    return np.max(np.abs(v1 - v2))

  else:
    raise ValueError("ERROR: Unknown distance type")

from typing import List, Tuple
import numpy as np

def mean(sorted_distance_target: List[Tuple[float, float]]) -> float:
  """
  Compute the mean of the target values from a list of tuples.

  This function takes a list of tuples, sorted according to the first elements of the tuples (i.e., the distance).

  Arguments:
    sorted_distance_target (List[Tuple[float, float]]): A list where each element is a tuple with two elements.
      The first element of each tuple is a distance.
      The second element of each tuple is a target value.
      The list is sorted in ascending order of the distance values.
  
  Returns:
    (float): The mean of the target values.
  """
  sum = 0

  for tp in sorted_distance_target:
    sum += tp[1]

  return sum / len(sorted_distance_target)

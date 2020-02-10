from numba import njit
import numpy as np

@njit
def np_apply_along_axis(func1d, arr, axis):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@njit
def allocating_items(dist_m, train_idxs, test_idxs, remaining_idxs, i):
    # 1 samples to Train
    r_of_remains = np.argmax(np_apply_along_axis(np.min, dist_m[:, train_idxs[:i + 2]][remaining_idxs[remaining_idxs != -1]], axis=1))
    r = remaining_idxs[remaining_idxs != -1][int(r_of_remains)]
    train_idxs[i + 2] = r
    remaining_idxs[r] = -1

    # 1 samples to Test
    r_of_remains = np.argmax(np_apply_along_axis(np.min, dist_m[:, test_idxs[:i + 2]][remaining_idxs[remaining_idxs != -1]], axis=1))
    r = remaining_idxs[remaining_idxs != -1][int(r_of_remains)]
    test_idxs[i + 2] = r
    remaining_idxs[r] = -1
    return train_idxs, test_idxs, remaining_idxs

import numpy as np

def normalize_array(arr, min_value=None, max_value=None):
    divider = (arr.max() - arr.min())
    if not divider:
        return np.zeros(arr.shape)
    normalized_array = (arr - arr.min()) / (arr.max() - arr.min())  # Normalize to 0-1
    if max_value or min_value:
        normalized_array = normalized_array * (max_value - min_value) + min_value  # Scale to min_value-max_value
    return normalized_array

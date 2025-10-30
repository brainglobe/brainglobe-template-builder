import numpy as np
from scipy.ndimage import generic_filter


def modal_filter_ignore_zeros(window):
    """
    Compute the mode of the window, ignoring zero values.

    Parameters
    ----------
    window : numpy.ndarray
        The input window of values.

    Returns
    -------
    int or float
        The most common non-zero value in the window, or 0 if all values
        are zero.
    """
    # Remove zeros from the window
    non_zero_values = window[window != 0]
    if len(non_zero_values) == 0:
        return 0  # If all values are zero, return 0
    # Compute the mode (most common value)
    values, counts = np.unique(non_zero_values, return_counts=True)
    return values[np.argmax(counts)]


def apply_modal_filter(image, filter_size=3):
    """Apply a modal filter to the image, ignoring zero neighbors.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a 3D NumPy array.
    filter_size : int
        Size of the filtering window (must be odd).

    Returns
    -------
    numpy.ndarray
        Filtered image.
    """
    # Apply the modal filter using a sliding window
    filtered_image = generic_filter(
        image, function=modal_filter_ignore_zeros, size=filter_size
    )
    return filtered_image

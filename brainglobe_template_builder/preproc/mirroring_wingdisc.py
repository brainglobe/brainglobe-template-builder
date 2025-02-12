import numpy as np


def mirroring (
    image_array: np.ndarray,
) -> np.ndarray:
    """Create a Mirror image of a 3D image along the (1,0) plane.

    Parameters
    ----------
    image_array : np.ndarray
        Array for image to mirror.
    """
    z, x, y = image_array.shape
    mirrored_image = np.zeros((z, x, y))
    for k in range(0, z):
        for i in range(0, x):
            for j in range(0, y):
                mirrored_image[k, -(i + 1), j] = image_array[k, i, j]
    return mirrored_image


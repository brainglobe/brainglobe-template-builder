import numpy as np


def mirroring (
    image_array: np.ndarray,
    axis: int
) -> np.ndarray:
    """Create a Mirror image of a 3D image along the specified axes.

    Parameters
    ----------
    image_array : np.ndarray
        Array for image to mirror.
    axis : int
    """
    mirrored_image =np.flip(image_array, axis=axis)
    return mirrored_image


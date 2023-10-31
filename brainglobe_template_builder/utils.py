from typing import Literal, Union

import numpy as np
from skimage import filters, measure


def extract_largest_object(binary_image):
    """Keep only the largest object in a binary image.

    Parameters
    ----------
    binary_image : np.ndarray
        A binary image.

    Returns
    -------
    np.ndarray
        A binary image containing only the largest object.
    """
    labeled_image = measure.label(binary_image)
    regions = measure.regionprops(labeled_image)
    largest_region = max(regions, key=lambda region: region.area)
    return labeled_image == largest_region.label


def threshold_image(
    image: np.ndarray,
    method: Literal["triangle", "otsu", "isodata"] = "triangle",
) -> Union[np.ndarray, None]:
    """Threshold an image using the specified method to get a binary mask.

    Parameters
    ----------
    image : np.ndarray
        Image to threshold.
    method : str
        Thresholding method to use. One of 'triangle', 'otsu', and 'isodata'
        (corresponding to methods from the skimage.filters module).
        Defaults to 'triangle'.

    Returns
    -------
    np.ndarray
        A binary mask.
    """

    method_to_func = {
        "triangle": filters.threshold_triangle,
        "otsu": filters.threshold_otsu,
        "isodata": filters.threshold_isodata,
    }
    if method in method_to_func.keys():
        thresholded = method_to_func[method](image)
        return image > thresholded
    else:
        raise ValueError(f"Unknown thresholding method {method}")

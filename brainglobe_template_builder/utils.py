from itertools import product
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


def get_midline_points(mask: np.ndarray):
    """Get a set of 9 points roughly on the x axis midline of a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask of shape (z, y, x).

    Returns
    -------
    np.ndarray
        An array of shape (9, 3) containing the midline points.
    """

    # Ensure mask is 3D
    if mask.ndim != 3:
        raise ValueError("Mask must be 3D")

    # Ensure mask is binary
    try:
        mask = mask.astype(bool)
    except ValueError:
        raise ValueError("Mask must be binary")

    # Derive mask properties
    props = measure.regionprops(measure.label(mask))[0]
    # bbox in shape (3, 2): for each dim (row) the min and max (col)
    bbox = np.array(props.bbox).reshape(2, 3).T
    bbox_ranges = bbox[:, 1] - bbox[:, 0]
    # mask centroid in shape (3,)
    centroid = np.array(props.centroid)

    # Find slices at 1/4, 2/4, and 3/4 of the z and y dimensions
    z_slices = [bbox_ranges[0] / 4 * i for i in [1, 2, 3]]
    y_slices = [bbox_ranges[1] / 4 * i for i in [1, 2, 3]]
    # Find points at the intersection the centroid's x slice
    # with the above y and z slices.
    # This produces a set of 9 points roughly on the midline
    points = list(product(z_slices, y_slices, [centroid[2]]))

    return np.array(points)


def fit_plane_to_points(
    points: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit a plane to a set of 3D points.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 3) containing the points.

    Returns
    -------
    np.ndarray
        The normal vector to the plane, with shape (3,).
    """

    # Ensure points are 3D
    if points.shape[1] != 3:
        raise ValueError("Points array must have 3 columns (z, y, x)")

    centered_points = points - np.mean(points, axis=0)

    # Use SVD to get the normal vector to the plane
    _, _, vh = np.linalg.svd(centered_points)
    normal_vector = vh[-1]

    return normal_vector


def align_vectors(v1, v2):
    """Align two vectors using Rodrigues' rotation formula.

    Parameters
    ----------
    v1 : np.ndarray
        The first vector.
    v2 : np.ndarray
        The second vector.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)
    s = np.linalg.norm(cross_prod)
    K = np.array(
        [
            [0, -cross_prod[2], cross_prod[1]],
            [cross_prod[2], 0, -cross_prod[0]],
            [-cross_prod[1], cross_prod[0], 0],
        ]
    )
    rotation = np.eye(3) + K + K @ K * ((1 - dot_prod) / (s**2))
    return rotation

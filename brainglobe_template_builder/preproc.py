from itertools import product
from typing import Literal, Union

import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from skimage import filters, measure, morphology


def _extract_largest_object(binary_image):
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


def _threshold_image(
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


def create_mask(
    image: np.ndarray,
    gauss_sigma: float = 3,
    threshold_method: Literal["triangle", "otsu", "isodata"] = "triangle",
    erosion_size: int = 5,
) -> np.ndarray:
    """Threshold image and create a mask for the largest object.

    The mask is generated by applying a Gaussian filter to the image,
    thresholding the smoothed image, keeping only the largest object, and
    eroding the resulting mask.

    Parameters
    ----------
    image : np.ndarray
        A 3D image to generate the mask from.
    gauss_sigma : float
        Standard deviation for Gaussian kernel (in pixels) to smooth image
        before thresholding. Set to 0 to skip smoothing.
    threshold_method : str
        Thresholding method to use. One of 'triangle', 'otsu', and 'isodata'
        (corresponding to methods from the skimage.filters module).
        Defaults to 'triangle'.
    erosion_size : int
        Size of the erosion footprint (in pixels) to apply to the mask.
        Set to 0 to skip erosion.

    Returns
    -------
    mask : np.ndarray
        A binary mask of the largest object in the image.
    """

    # Check input
    if image.ndim != 3:
        raise ValueError("Image must be 3D")

    if gauss_sigma > 0:
        data_smoothed = filters.gaussian(image, sigma=gauss_sigma)
    else:
        data_smoothed = image

    binary = _threshold_image(data_smoothed, method=threshold_method)
    mask = _extract_largest_object(binary)

    if erosion_size > 0:
        mask = morphology.binary_erosion(
            mask, footprint=np.ones((erosion_size,) * image.ndim)
        )
    return mask


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

    # Check input
    if mask.ndim != 3:
        raise ValueError("Mask must be 3D")

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


def _fit_plane_to_points(
    points: np.ndarray,
) -> np.ndarray:
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


def align_to_midline(
    image: np.ndarray,
    points: np.ndarray,
    axis: Literal["x", "y", "z"] = "x",
) -> np.ndarray:
    """Transform image such that the midline of the specified axis is aligned
    with the plane fitted to the provided points.

    This function first fits a plane to the points, then rigidly transforms
    the image such that the fitted plane is aligned with the axis midline.

    Parameters
    ----------
    image : np.ndarray
        A 3D image to align.
    points : np.ndarray
        An array of shape (n_points, 3) containing points.
    axis : str
        Axis to align the midline with. One of 'x', 'y', and 'z'.
        Defaults to 'x'.

    Returns
    -------
    aligned_image : np.ndarray
        A 3D array containing the transformed image.
    """

    # Check input
    if image.ndim != 3:
        raise ValueError("Image must be 3D")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be an array of shape (n_points, 3)")
    if axis not in ["x", "y", "z"]:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")

    # Fit a plane to the points
    normal_vector = _fit_plane_to_points(points)

    # Compute centroid of the midline points
    centroid = np.mean(points, axis=0)

    # Translation of the centroid to the origin
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -centroid

    # Rotation to align normal vector with unit vector along the specified axis
    axis_vec = np.zeros(3)
    axis_index = {"z": 0, "y": 1, "x": 2}[axis]  # axis order is zyx in napari
    axis_vec[axis_index] = 1
    rotation_to_axis = Rotation.align_vectors(
        axis_vec.reshape(1, 3),
        normal_vector.reshape(1, 3),
    )[0].as_matrix()
    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation_to_axis

    # Translation back, so that the plane is in the middle of axis
    translation_to_mid_axis = np.eye(4)
    translation_to_mid_axis[axis_index, 3] = (
        image.data.shape[axis_index] // 2 - centroid[axis_index]
    )

    # Combine transformations into a single 4x4 matrix
    transformation_matrix = (
        np.linalg.inv(translation_to_origin)
        @ rotation_4x4
        @ translation_to_origin
        @ translation_to_mid_axis
    )

    # Apply the transformation to the image
    aligned_image = affine_transform(
        image,
        transformation_matrix[:3, :3],
        offset=transformation_matrix[:3, 3],
    )
    return aligned_image

from itertools import product
from typing import Literal

import numpy as np
from scipy.ndimage import affine_transform
from skimage import measure


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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane to a set of 3D points.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n_points, 3) containing the points.

    Returns
    -------
    centroid : np.ndarray
        The centroid of the points.
    normal_vector : np.ndarray
        A vector normal to the fitted plane.
    """

    # Find the centroid of the points
    centroid = np.mean(points, axis=0)
    # Use SVD to get the normal vector to the plane
    _, _, vh = np.linalg.svd(points - centroid)
    normal_vector = vh[-1]

    return centroid, normal_vector


def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray):
    """Find the rotation matrix that aligns vec1 to vec2. Implementation
    adapted from StackOverflow [1]_.

    Parameters
    ----------
    vec1 : np.ndarray
        The 3D "source" vector
    vec2 : np.ndarray
        The 3D "target" vector

    Returns
    -------
    A rotation matrix (3x3) that, when applied to vec1, aligns it with vec2.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/45142959
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def get_alignment_transform(
    image: np.ndarray,
    points: np.ndarray,
    axis: Literal["x", "y", "z"] = "x",
) -> np.ndarray:
    """Find the transformation matrix that aligns the plane defined by the
    given points to the midline of the specified axis.

    Parameters
    ----------
    image : np.ndarray
        A 3D image to align.
    points : np.ndarray
        An array of shape (n_points, 3) containing points.
    axis : str
        Axis to align the midline with. One of 'x', 'y', and 'z'.
        Defaults to 'x'. The axis order is zyx in napari.

    Returns
    -------
    transform: np.ndarray
        A 4x4 rigid transformation matrix (3x3 rotation matrix with a 3x1
        translation vector appended to the right).
    """

    # Check input
    if image.ndim != 3:
        raise ValueError("Image must be 3D")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be an array of shape (n_points, 3)")
    if axis not in ["x", "y", "z"]:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")

    # Fit a plane to the points
    centroid, normal_vector = _fit_plane_to_points(points)

    # Construct a unit vector along the specified axis
    axis_idx = {"z": 0, "y": 1, "x": 2}[axis]  # axis order is zyx in napari
    axis_vector = np.zeros(3)
    axis_vector[axis_idx] = 1

    # invert the normal vector if it points in the opposite direction of the
    # specified axis
    if np.dot(normal_vector, axis_vector) < 0:
        normal_vector = -normal_vector

    # Compute the necessary transforms
    # 1. translate to origin (so that centroid is at origin)
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -centroid
    # 2. rotate to align fitted plane with specified axis
    rotation = np.eye(4)
    rotation[:3, :3] = _rotation_matrix_from_vectors(
        normal_vector, axis_vector
    )
    # 3. translate to mid-axis (so that centroid is at middle of axis)
    translation_to_mid_axis = np.eye(4)
    offset = (image.shape[axis_idx] / 2 - centroid[axis_idx]) * axis_vector
    translation_to_mid_axis[:3, 3] = centroid + offset
    # Combine the transforms
    combined_transform = (
        translation_to_mid_axis @ rotation @ translation_to_origin
    )
    return combined_transform


def apply_transform(
    data: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transformation to an image.

    Parameters
    ----------
    data : np.ndarray
        A 3D image to transform.
    transform : np.ndarray
        A 4x4 transformation matrix.

    Returns
    -------
    np.ndarray
        The transformed data.

    Notes
    -----
    This function inverts the affine and flips the offset when passing the data
    to `scipy.ndimage.affine_transform`. This is because the transforms are
    given in the 'push' (or 'forward') direction, transforming input to output,
    whereas `scipy.ndimage.affine_transform` does `pull` (or `backward`)
    resampling, transforming the output space to the input.
    """

    if data.ndim != 3:
        raise ValueError("Data must be 3D")
    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix")

    transformed = affine_transform(
        data,
        np.linalg.inv(transform[:3, :3]),
        offset=-transform[:3, 3],
    )
    return transformed

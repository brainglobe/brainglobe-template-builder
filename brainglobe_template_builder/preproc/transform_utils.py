import dask.array as da
import numpy as np
from scipy.ndimage import affine_transform
from skimage import transform


def get_rotation_from_vectors(vec1: np.ndarray, vec2: np.ndarray):
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
    This function inverts the affine and flips the offset when passing the
    data to `scipy.ndimage.affine_transform`. This is because the
    transforms are given in the 'forward' direction, transforming input
    to output, whereas `scipy.ndimage.affine_transform` does 'backward'
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
    # Preserve original data range and type
    transformed = np.clip(transformed, data.min(), data.max())
    return transformed.astype(data.dtype)


def downsample_anisotropic_image_stack(
    stack: da.Array, xy_downsampling: int, z_downsampling: int
) -> np.ndarray:
    """

    Lazily downsamples a dask array first along axis 1,2 and then along axis 0,
    using a local mean of the pixels. The (smaller) array is returned
    in memory (numpy) form at the end.

    This setup is typical for certain types of microscopy,
    where z-resolution is lower than x-y-resolution.

    The input dask array must be chunked by x-y slice,

    Parameters:
    ----------
    stack : da.Array
        The input dask array representing the image stack.
    xy_downsampling : int
        The downsampling factor for the x and y axes.
    z_downsampling : int
        The downsampling factor for the z axis.
    Returns:
    -------
    np.ndarray
        The computed downsampled (numpy) array.
    Raises:
    ------
    AssertionError
        If the array is not chunked slice-wise on axis 0.
    """
    # check we have expected slice chunks
    assert np.all(
        np.array(stack.chunks[0]) == 1
    ), f"Array not chunked slice-wise! Chunks on axis 0 are {stack.chunks[0]}"

    # we have xy slices as chunks, so apply downscaling in xy first
    downsampled_xy = stack.map_blocks(
        transform.downscale_local_mean,
        (1, xy_downsampling, xy_downsampling),
        dtype=np.float64,
    )

    # rechunk so we can map_blocks along z
    downsampled_xy = downsampled_xy.rechunk(
        {0: downsampled_xy.shape[0], 1: -1, 2: -1}
    )

    # downsample in z
    downsampled_z = downsampled_xy.map_blocks(
        transform.downscale_local_mean,
        (z_downsampling, 1, 1),
        dtype=np.float64,
    )
    return downsampled_z.compute()
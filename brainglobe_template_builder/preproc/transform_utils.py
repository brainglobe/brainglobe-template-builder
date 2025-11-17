from pathlib import Path

import dask.array as da
import numpy as np
from brainglobe_utils.IO.image import read_z_stack, save_any
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
    This function inverts the affine and when passing the
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
        np.linalg.inv(transform),
    )
    # Preserve original data range and type
    transformed = np.clip(transformed, data.min(), data.max())
    return transformed.astype(data.dtype)


def downsample_anisotropic_image_stack(
    stack: da.Array, in_plane_factor: int, axial_factor: int
) -> np.ndarray:
    """

    Lazily downsamples a dask array first along axes 1,2 (in-plane) and then
    along axis 0 (axial), using a local mean of the pixels. The image is
    zero-padded to allow for the correct dimensions of the averaging
    neighbourhood, since it uses `skimage.transform.downscale_local_mean`
    under the hood.

    This setup is typical for certain types of microscopy,
    where axial resolution is lower than in-plane resolution.

    The input dask array must be chunked by plane. The (smaller) array
    is returned in memory (numpy) form at the end.

    Parameters:
    ----------
    stack : da.Array
        The input dask array representing the image stack.
    in_plane_factor : int
        The in-plane downsampling factor (axes=1,2).
    axial_factor : int
        The downsampling factor in axial direction (axis=0).
    Returns:
    -------
    np.ndarray
        The computed downsampled (numpy) array.
    Raises:
    ------
    AssertionError
        If the array is not chunked by plane along axis 0.
    """
    # check we have expected slice chunks
    assert np.all(
        np.array(stack.chunks[0]) == 1
    ), f"Array not chunked by plane! Chunks on axis 0 are {stack.chunks[0]}"

    # we have xy slices as chunks, so apply downscaling in xy first
    downsampled_inplane = stack.map_blocks(
        transform.downscale_local_mean,  # type: ignore
        (1, in_plane_factor, in_plane_factor),
        dtype=np.float64,
    )

    # rechunk so we can map_blocks along z
    downsampled_inplane = downsampled_inplane.rechunk(
        {0: downsampled_inplane.shape[0], 1: -1, 2: -1}
    )

    # downsample in z
    downsampled_axial = downsampled_inplane.map_blocks(
        transform.downscale_local_mean,
        (axial_factor, 1, 1),
        dtype=np.float64,
    )
    return downsampled_axial.compute()


def _downsample_anisotropic_stack_by_factors(
    stack: da.Array, downsampling_factors: list[float]
) -> np.ndarray:

    # check we have expected slice chunks
    if not np.all(np.array(stack.chunks[0]) == 1):
        raise ValueError(
            "Array not chunked by plane! Chunks on "
            f"axis 0 are {stack.chunks[0]}"
        )

    # we have xy slices as chunks, so apply downscaling in xy first
    downsampled_inplane = stack.map_blocks(
        transform.rescale,
        (1, downsampling_factors[1], downsampling_factors[2]),
        dtype=np.float64,
    )

    # rechunk so we can map_blocks along z
    downsampled_inplane = downsampled_inplane.rechunk(
        {0: downsampled_inplane.shape[0], 1: -1, 2: -1}
    )

    # downsample in z
    downsampled_axial = downsampled_inplane.map_blocks(
        transform.rescale,
        (downsampling_factors[0], 1, 1),
        dtype=np.float64,
    )
    return downsampled_axial.compute()


def downsample_anisotropic_stack_to_isotropic(
    stack: da.Array, input_vox_sizes: list[float], output_vox_size: float
) -> np.ndarray:

    # Don't allow up-sampling
    for vox_size in input_vox_sizes:
        if vox_size > output_vox_size:
            raise ValueError(
                f"Some input voxel sizes: {input_vox_sizes} are larger "
                f"than the output_vox_size: {output_vox_size}. "
                "Upsampling would be required."
            )

    downsampling_factors = [
        vox_size / output_vox_size for vox_size in input_vox_sizes
    ]
    downsampled_image = _downsample_anisotropic_stack_by_factors(
        stack, downsampling_factors
    )

    # if the shape of each axis isn't exactly divisible by its
    # downsampling factor, the final shape
    # (and therefore voxel size) will be slightly off the target
    # output_vox_size
    # for axis_shape, downsampling_factor in
    # zip(stack.shape, downsampling_factors):
    #     if axis_shape % downsampling_factor != 0:

    return downsampled_image


def downsample(
    sample_folder: Path,
    downsampled_path: Path,
    in_plane_factor: int,
    axial_factor: int,
) -> None:
    """Convenience function to read, downsample and write
    an anisotropic stack of images"""
    stack = read_z_stack(str(sample_folder))

    downsampled = downsample_anisotropic_image_stack(
        stack, in_plane_factor=in_plane_factor, axial_factor=axial_factor
    )
    save_any(downsampled, downsampled_path)

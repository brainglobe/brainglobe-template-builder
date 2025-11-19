from pathlib import Path

import dask.array as da
import numpy as np
from brainglobe_utils.IO.image import read_with_dask, save_any
from loguru import logger
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


def _verify_chunked_by_entire_plane(stack: da.Array) -> None:
    """Check the dask array's chunks cover entire z slices."""

    expected_chunk_size = (1, stack.shape[1], stack.shape[2])
    for i, chunk_size in enumerate(expected_chunk_size):
        if not (np.array(stack.chunks[i]) == chunk_size).all():
            raise ValueError(
                "Array not chunked by entire plane! Chunks on "
                f"axis {i} are {stack.chunks[i]}"
            )


def _downsample_anisotropic_stack_by_factors(
    stack: da.Array, downsampling_factors: list[float]
) -> np.ndarray:
    """
    Lazily downsamples a dask array first along axes 1,2 (in-plane) and then
    along axis 0 (axial), using interpolation via `skimage.transform.rescale`.

    The input dask array must be chunked by the entire plane. The (smaller)
    array is returned in memory (numpy) form at the end.


    Parameters
    ----------
    stack : da.Array
        The input dask array representing the image stack.
    downsampling_factors : list[float]
        Downsampling factors to use for each image axis. They should be
        defined so that: input shape * factor = output shape so e.g. a
        0.5 downsampling factor, would be equivalent to downsampling 2x.

    Returns
    -------
    np.ndarray
        The computed downsampled (numpy) array.

    Raises
    ------
    ValueError
        If the array is not chunked by plane along axis 0.
    """

    # check we have expected slice chunks
    _verify_chunked_by_entire_plane(stack)

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


def _warn_if_output_vox_sizes_incorrect(
    input_shape: tuple[int],
    output_shape: tuple[int],
    input_vox_sizes: list[float],
    downsampling_factors: list[float],
) -> None:
    """Log a warning if the output voxel sizes are incorrect.

    If the shape of each image axis isn't exactly divisible by its
    downsampling factor, the final shape (and therefore voxel size)
    will be slightly different from the target voxel size.

    Parameters
    ----------
    input_shape : tuple[int]
        Shape of input image (before downsampling).
    output_shape : tuple[int]
        Shape of output image (after downsampling).
    input_vox_sizes : list[float]
        Input image voxel sizes.
    downsampling_factors : list[float]
        Downsampling factors used for each axis.
    """

    for axis_shape, downsampling_factor in zip(
        input_shape, downsampling_factors
    ):
        if not (axis_shape * downsampling_factor).is_integer():

            # Total size of each axis in microns
            axis_sizes_um = [
                axis_shape * vox_size
                for axis_shape, vox_size in zip(input_shape, input_vox_sizes)
            ]
            # output voxel sizes in microns
            output_vox_sizes = [
                axis_size / axis_shape
                for axis_size, axis_shape in zip(axis_sizes_um, output_shape)
            ]

            msg = (
                f"Image shape {input_shape} isn't an exact multiple of "
                f"downsampling factors [{1/downsampling_factors[0]:.3f}x, "
                f"{1/downsampling_factors[1]:.3f}x "
                f"{1/downsampling_factors[2]:.3f}x]. "
                f"Output shape is {output_shape}, with voxel size "
                f"[{output_vox_sizes[0]:.3f}, {output_vox_sizes[1]:.3f}, "
                f"{output_vox_sizes[2]:.3f}]."
            )
            logger.warning(msg)
            return


def downsample_anisotropic_stack_to_isotropic(
    stack: da.Array, input_vox_sizes: list[float], output_vox_size: float
) -> np.ndarray:
    """
    Lazily downsamples a dask array first along axes 1,2 (in-plane) and then
    along axis 0 (axial), using interpolation via `skimage.transform.rescale`.

    This setup is typical for certain types of microscopy,
    where axial resolution is lower than in-plane resolution.

    The input dask array must be chunked by the entire plane. The (smaller)
    array is returned in memory (numpy) form at the end.


    Parameters
    ----------
    stack : da.Array
        The input dask array representing the image stack.
    input_vox_sizes : list[float]
        Input voxel sizes in microns (must be in same order as stack axes).
    output_vox_size : float
        Output voxel size in microns to downsample to. The image will
        be made isotropic, with voxel sizes as close as possible to
        [output_vox_size, output_vox_size, output_vox_size].

    Returns
    -------
    np.ndarray
        The computed downsampled (numpy) array.

    Raises
    ------
    ValueError
        If the array is not chunked by plane along axis 0, or
        if the output_vox_size would require upsampling.
    """

    # Don't allow up-sampling
    for vox_size in input_vox_sizes:
        if vox_size > output_vox_size:
            raise ValueError(
                f"Some input voxel sizes: {input_vox_sizes} are larger "
                f"than the output_vox_size: {output_vox_size}. "
                "Upsampling would be required."
            )

    # Downsampling factors are defined so: input shape * factor = output shape
    # so e.g. a 0.5 downsampling factor, is equivalent to downsampling 2x
    downsampling_factors = [
        vox_size / output_vox_size for vox_size in input_vox_sizes
    ]
    downsampled_image = _downsample_anisotropic_stack_by_factors(
        stack, downsampling_factors
    )

    _warn_if_output_vox_sizes_incorrect(
        stack.shape,
        downsampled_image.shape,
        input_vox_sizes,
        downsampling_factors,
    )

    return downsampled_image


def downsample(
    sample_folder: Path,
    downsampled_path: Path,
    downsampling_factors: list[float],
) -> None:
    """Convenience function to read, downsample and write
    an anisotropic stack of images"""
    stack = read_with_dask(str(sample_folder))

    downsampled = _downsample_anisotropic_stack_by_factors(
        stack, downsampling_factors
    )
    save_any(downsampled, downsampled_path)

import dask.array as da
import numpy as np
from skimage import transform


def resize_anisotropic_image_stack(
    stack: da.Array, in_plane_factor: float, axial_factor: float
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
        transform.rescale,  # type: ignore
        (1, in_plane_factor, in_plane_factor),
        dtype=np.float64,
    )

    # rechunk so we can map_blocks along z
    downsampled_inplane = downsampled_inplane.rechunk(
        {0: downsampled_inplane.shape[0], 1: -1, 2: -1}
    )

    # downsample in z
    downsampled_axial = downsampled_inplane.map_blocks(
        transform.rescale,
        (axial_factor, 1, 1),
        dtype=np.float64,
    )
    return downsampled_axial.compute()


def normalize_planes_by_mean(image: np.ndarray, percentile=0.1) -> np.ndarray:
    """
    Parameters:
        image (np.ndarray): 3D numpy array (Z, Y, X).

    Returns:
        np.ndarray: Normalized 3D image.
    """
    higher_signal_threshold = (1 - percentile) * np.max(image)
    lower_signal_threshold = percentile * np.max(image)
    mask = (image > lower_signal_threshold) & (image < higher_signal_threshold)
    normalized = image.copy().astype(np.float32)
    overall_mean = np.mean(image[mask])
    for i in range(image.shape[0]):
        plane = image[i]
        if np.any(mask[i]):
            mean_val = plane[mask[i]].mean()
            normalized[i] = plane.astype(np.float32) / mean_val
            normalized[i] *= overall_mean
            normalized[i] = np.clip(
                normalized[i], 0, 65535
            )  # Ensure values are within uint16 range
        else:
            normalized[i] = plane
    return normalized.astype(np.uint16)

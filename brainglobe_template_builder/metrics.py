"""Useful metrics for image quality assessment."""

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    gaussian_gradient_magnitude,
    laplace,
)
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import ball

EPS: float = 1e-12  # Small constant to avoid division by zero


def smooth(image: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Smooth an image volume using a Gaussian filter."""
    return gaussian_filter(image, sigma=sigma, mode="constant")


def laplacian(image: np.ndarray) -> float:
    """Calculate the the Laplacian of an image volume."""
    return laplace(image, mode="constant")


def gradient_magnitude(
    image: np.ndarray, sigma: float = 0.0, normalize: bool = False
) -> np.ndarray:
    """Calculate the gradient magnitude of an image volume.

    Gaussian smoothing is applied before computing the gradient.

    Parameters
    ----------
    image: numpy.ndarray
        Input image volume.
    sigma: float
        Standard deviation for Gaussian filter.
        Default is 0.0, meaning no smoothing is applied.
    normalize: bool
        If True, the gradient magnitude is normalized by the smoothed image.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Gradient magnitude image.
    """
    gmag = gaussian_gradient_magnitude(image, sigma=sigma, mode="constant")
    if normalize:
        image_smooth = smooth(image, sigma=sigma)
        gmag = gmag / (np.abs(image_smooth) + EPS)
    return gmag


def create_edge_mask_3d(
    image: np.ndarray,
    sigma: float = 1.0,
    threshold: float | None = None,
    dilate_radius: int | None = 1,
) -> np.ndarray:
    """Create a binary edge mask for an image volume using a Sobel filter.

    The image is first smoothed with a Gaussian filter,
    then the Sobel filter is applied to detect edges, and finally
    a binary mask is created based on Otsu's method or a provided
    threshold.

    Parameters
    ----------
    image: np.ndarray
        Input image volume.
    sigma: float
        Standard deviation for Gaussian filter.
        Default is 1.0.
    threshold: float | None
        Threshold for edge detection, applied to the Sobel-filtered image.
        If None, Otsu's method is used to determine the threshold.
    dilate_radius: int
        Radius for dilation of the edge mask.
        Default is 1.

    Returns
    -------
    np.ndarray
        Binary edge mask.

    """
    image_smooth = smooth(image, sigma=sigma)
    edge_strength = sobel(image_smooth, mode="constant")

    if threshold is None:
        threshold = threshold_otsu(edge_strength)

    edge_mask = edge_strength > threshold

    if dilate_radius is not None:
        structure = ball(max(1, int(dilate_radius))).astype(bool)
        edge_mask = binary_dilation(edge_mask, structure=structure)

    return edge_mask


def edge_snr_3d(
    grad_mag: np.ndarray,
    edge_mask: np.ndarray,
    non_edge_mask: np.ndarray | None = None,
) -> float:
    """Compute edge SNR for 3D image volumes.

    Edge SNR is defined as the ratio of the mean gradient magnitude on edges
    to the standard deviation of the gradient magnitude on non-edges.

    Parameters
    ----------
    grad_mag: np.ndarray
        Gradient magnitude image.
    edge_mask: np.ndarray
        Binary mask indicating the edges in the image.
    non_edge_mask: np.ndarray
        Binary mask indicating the non-edges in the image.
        If None, the non-edges are defined as the complement of the edge mask.

    Returns
    -------
    float
        The edge SNR value.

    """
    edge_gmag = grad_mag[edge_mask]
    if non_edge_mask is None:
        non_edge_mask = ~edge_mask
    non_edge_gmag = grad_mag[non_edge_mask]

    if edge_gmag.size == 0 or non_edge_gmag.size == 0:
        return float("nan")

    return float(edge_gmag.mean() / (non_edge_gmag.std(ddof=0) + EPS))

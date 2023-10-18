from pathlib import Path
import numpy as np
import imio


def save_nii(stack: np.ndarray, pix_sizes: list, dest_path: Path):
    """
    Save 3D image stack to dest_path as a nifti image.

    Parameters
    ----------
    stack : np.ndarray
        3D image stack
    pix_sizes : list
        list of pixel dimensions in mm. The order is 'x', 'y', 'z'
    dest_path : pathlib.Path
        path to save the nifti image
    """
    transformation_matrix = _get_transf_matrix_from_res(pix_sizes)
    imio.to_nii(
        stack,
        dest_path.as_posix(),
        scale=pix_sizes,
        affine_transform=transformation_matrix,
    )


def _get_transf_matrix_from_res(pix_sizes: list) -> np.ndarray:
    """Create transformation matrix from a dictionary of pixel dimensions.

    Parameters
    ----------
    pix_sizes : list
        list of pixel dimensions in mm. The order is 'x', 'y', 'z'

    Returns
    -------
    np.ndarray
        A (4, 4) transformation matrix with the pixel dimensions on the diagonal
    """
    transformation_matrix = np.eye(4)
    for i in range(3):
        transformation_matrix[i, i] = pix_sizes[i]
    return transformation_matrix


def tiff_to_nifti(tiff_path: Path, nifti_path: Path, pix_sizes: list):
    """
    Convert a tiff image to a nifti image

    Parameters
    ----------
    tiff_path : pathlib.Path
        path to the tiff image
    nifti_path : pathlib.Path
        path to save the nifti image
    pix_sizes : list
        list of pixel dimensions in mm. The order is 'x', 'y', 'z'
    """
    stack = imio.load_any(tiff_path.as_posix())
    save_nii(stack, pix_sizes, nifti_path.as_posix())


def nifti_to_tiff(nifti_path: Path, tiff_path: Path):
    """
    Convert a nifti image to a tiff image

    Parameters
    ----------
    nifti_path : pathlib.Path
        path to the nifti image
    tiff_path : pathlib.Path
        path to save the tiff image
    """
    stack = imio.load_any(nifti_path.as_posix())
    imio.to_tiff(stack, tiff_path.as_posix())

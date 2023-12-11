import imio
import numpy as np


def save_nii(stack: np.ndarray, pix_sizes: list, dest_path: str):
    """
    Save stack to dest_path as a nifti image.
    :param stack: 3D numpy array
    :param pix_sizes: list of pixel sizes in mm
    :param str dest_path: Where to save the image on the filesystem
    """
    transformation_matrix = get_transf_matrix_from_res(pix_sizes)
    imio.to_nii(
        stack,
        dest_path,
        scale=pix_sizes,
        affine_transform=transformation_matrix,
    )


def get_transf_matrix_from_res(pix_sizes: list) -> np.ndarray:
    """Create transformation matrix
    from a dictionary of pixel sizes
    :param pix_sizes: list of pixel sizes in mm
    :return: transformation matrix in mm
    """
    transformation_matrix = np.eye(4)
    for i in [0, 1, 2]:
        transformation_matrix[i, i] = pix_sizes[i]
    return transformation_matrix


def tiff_to_nifti(tiff_path: str, nifti_path: str, pix_sizes: list):
    """
    Convert a tiff image to a nifti image
    :param tiff_path: path to the tiff image
    :param nifti_path: path to the nifti image
    :param pix_sizes: list of tiff pixel sizes in um
    """
    stack = imio.load_any(tiff_path)
    # convert pixel sizes from um to mm
    pix_sizes = [pix_size / 1000 for pix_size in pix_sizes]
    save_nii(stack, pix_sizes, nifti_path)


def nifti_to_tiff(nifti_path: str, tiff_path: str):
    """
    Convert a nifti image to a tiff image
    :param nifti_path: path to the nifti image
    :param tiff_path: path to the tiff image
    """
    stack = imio.load_any(nifti_path)
    imio.to_tiff(stack, tiff_path)

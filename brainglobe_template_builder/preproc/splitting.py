"""Utilities for splitting arrays into two halves along a symmetry axis."""

from pathlib import Path

import numpy as np

from brainglobe_template_builder.io import save_nii


def get_right_and_left_slices(array: np.ndarray) -> tuple:
    """
    Get the slices for equally splitting an array into right and left halves.

    Parameters
    ----------
    array : np.ndarray
        Array to split. Must be a 3D array in ASR orientation.

    Returns
    -------
    A tuple of slice objects for the right and left halves of the array.
    """
    slices = [slice(None)] * array.ndim
    axis = -1  # right-left axis is always the last in ASR orientation

    right_slices = slices.copy()
    right_slices[axis] = slice(0, array.shape[axis] // 2)

    left_slices = slices.copy()
    left_slices[axis] = slice(array.shape[axis] // 2, array.shape[axis])
    return tuple(right_slices), tuple(left_slices)


def generate_arrays_4template(
    brain: np.ndarray, mask: np.ndarray, pad: int = 0
) -> dict[str, np.ndarray]:
    """Generate all needed arrays for the template building process.

    Parameters
    ----------
    brain : np.ndarray
        The aligned brain image to split into hemispheres and symmetrise.
    mask : np.ndarray
        The aligned mask to split into hemispheres and symmetrise.
        Must be a binary mask and have the same shape as the image.
    pad : int, optional
        Number of planes to zero-pad arrays with. Default is 0 (no padding).
        The same number of planes will be added to all arrays, along each axis.
        This is useful to ensure that registration algorithms can correctly
        handle the array edges.

    Returns
    -------
    A dictionary containing the following arrays:
    - asym-brain: the input aligned image (asymmetric brain)
    - asym-mask: the input aligned mask (asymmetric mask)
    - right-hemi-brain: the right hemisphere of the image
    - right-hemi-mask: the right hemisphere of the mask
    - right-hemi-xflip-brain: the reflection of the right hemisphere
    - right-hemi-xflip-mask: the reflection of the right hemisphere mask
    - left-hemi-brain: the left hemisphere of the image
    - left-hemi-mask: the left hemisphere of the mask
    - left-hemi-xflip-brain: the reflection of the left hemisphere
    - left-hemi-xflip-mask: the reflection of the left hemisphere mask
    - right-sym-brain: the right hemisphere merged with its reflection
    - right-sym-mask: the right hemisphere mask merged with its reflection
    - left-sym-brain: the left hemisphere merged with its reflection
    - left-sym-mask: the left hemisphere mask merged with its reflection
    """

    # ensure mask in uint8
    mask = mask.astype(np.uint8)

    # ensure shapes match
    assert brain.ndim == 3
    assert brain.shape == mask.shape

    # Put the input arrays into the dictionary
    out_dict = {
        "asym-brain": brain,
        "asym-mask": mask,
    }

    right_half, left_half = get_right_and_left_slices(brain)

    for label, arr in zip(("brain", "mask"), (brain, mask)):
        right_half_arr = arr[right_half]
        left_half_arr = arr[left_half]
        right_half_arr_xflip = np.flip(right_half_arr, axis=-1)
        left_half_arr_xflip = np.flip(left_half_arr, axis=-1)
        right_sym_arr = np.dstack([right_half_arr, right_half_arr_xflip])
        left_sym_arr = np.dstack([left_half_arr_xflip, left_half_arr])
        out_dict.update(
            {
                f"right-hemi-{label}": right_half_arr,
                f"right-hemi-xflip-{label}": right_half_arr_xflip,
                f"left-hemi-{label}": left_half_arr,
                f"left-hemi-xflip-{label}": left_half_arr_xflip,
                f"right-sym-{label}": right_sym_arr,
                f"left-sym-{label}": left_sym_arr,
            }
        )

    if pad > 0:
        for key, arr in out_dict.items():
            out_dict[key] = np.pad(arr, pad_width=pad, mode="constant")

    return out_dict


def save_array_dict_to_nii(
    array_dict: dict[str, np.ndarray],
    save_dir: Path,
    vox_sizes: list[float],
):
    """Save arrays in a dictionary to NIfTI files in the specified directory.

    Parameters
    ----------
    array_dict : dict
        A dictionary containing numpy arrays to save. The keys should be the
        filenames (without extension) and the values should be the arrays.
        If the arrays are masks, the keys should contain the string "mask".
    save_dir : Path
        Directory to save the files to.
    vox_sizes : tuple
        Tuple of voxel sizes in mm.
    """

    assert save_dir.is_dir(), f"Directory {save_dir} does not exist."

    for key, data in array_dict.items():
        file_path = save_dir / f"{key}.nii.gz"
        if "mask" in key:
            save_nii(data, vox_sizes, file_path, kind="label")
        else:
            save_nii(data, vox_sizes, file_path, kind="image")

from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image.load import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii, to_tiff


def get_unique_folder_in_dir(search_dir: Path, search_str: str) -> Path:
    """
    Find a folder in a directory that contains a unique string.

    Parameters
    ----------
    search_dir : Path
        Directory to search in
    search_str : str
        String to search for in folder names

    Returns
    -------
    folder : Path
        Path to the folder that contains the search string
    """
    all_folders = [x for x in search_dir.iterdir() if x.is_dir()]
    folders_with_str = [x for x in all_folders if search_str in x.name]
    if len(folders_with_str) == 0:
        raise ValueError(f"No folders with {search_str} found")
    if len(folders_with_str) > 1:
        raise ValueError(f"Multiple folders with {search_str} found")
    return folders_with_str[0]


def load_tiff(tiff_path: Path):
    """
    Load an image to napari

    Parameters
    ----------
    tiff_path : pathlib.Path
        path to the tiff image

    Returns
    -------
    image : np.ndarray
    """
    valid_extensions = [".tif", ".tiff"]
    if tiff_path.suffix not in valid_extensions:
        raise ValueError(
            f"File extension {tiff_path.suffix} is not valid. "
            f"Expected one of {valid_extensions}"
        )
    image = load_any(tiff_path.as_posix())
    return image


def save_3d_points_to_csv(points: np.ndarray, file_path: Path):
    """
    Save 3D points to a csv file
    """

    if points.shape[1] != 3:
        raise ValueError(
            f"Points must be of shape (n, 3). Got shape {points.shape}"
        )
    if file_path.suffix != ".csv":
        raise ValueError(
            f"File extension {file_path.suffix} is not valid. "
            f"Expected file path to end in .csv"
        )

    points_df = pd.DataFrame(points, columns=["z", "y", "x"])
    points_df.to_csv(file_path, index=False)


def file_path_with_suffix(path: Path, suffix: str, new_ext=None) -> Path:
    """
    Return a new path with the given suffix added before the extension.

    "suffix" the string to add before the left-most period, while
    extension is the string after that. For example, if ``suffix=="_new"``,
    and ``new_ext=".nii.gz"``, the output path will end with "_new.nii.gz".

    Parameters
    ----------
    path : pathlib.Path
        The file path to modify.
    suffix : str
        The suffix to add before the extension.
    new_ext : pathlib.Path, optional
        If given, replace the current extension with this one.
        Should include the leading period.

    Returns
    -------
    Path
        The new path to the file with the given suffix.

    """
    suffixes = "".join(path.suffixes)
    pure_stem = str(path.stem).rstrip(suffixes)
    if new_ext is not None:
        new_name = f"{pure_stem}{suffix}{new_ext}"
    else:
        new_name = f"{pure_stem}{suffix}{suffixes}"
    return path.with_name(new_name)


def tiff_to_nifti(tiff_path: Path, nifti_path: Path, vox_sizes: list):
    """
    Convert a tiff image to a nifti image

    Parameters
    ----------
    tiff_path : pathlib.Path
        path to the tiff image
    nifti_path : pathlib.Path
        path to save the nifti image
    vox_sizes : list
        list of voxel dimensions in mm, in the same order as the image axes.
    """
    stack = load_any(tiff_path.as_posix())
    save_as_asr_nii(stack, vox_sizes, nifti_path)


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
    stack = load_any(nifti_path.as_posix())
    to_tiff(stack, tiff_path.as_posix())

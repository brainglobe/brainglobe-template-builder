from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from brainglobe_utils.IO.image.load import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.plots import plot_grid
from brainglobe_template_builder.preproc.brightness import (
    correct_image_brightness,
)
from brainglobe_template_builder.preproc.cropping import crop_to_mask
from brainglobe_template_builder.preproc.masking import create_mask
from brainglobe_template_builder.preproc.preproc_config import PreprocConfig
from brainglobe_template_builder.validate import validate_input_csv

DERIVATIVES_DIR_NAME = "derivatives"


def _create_subject_dir(subject_id: str, output_dir: Path) -> Path:
    """Create subject dir inside of the derivatives directory."""
    subject_dir = output_dir / DERIVATIVES_DIR_NAME / f"sub-{subject_id}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    return subject_dir


def _save_niftis(
    image: np.ndarray,
    voxel_sizes_mm: list[float],
    subject_dir: Path,
    image_name: str,
) -> tuple[Path, Path]:
    """Save image as nifti files - in standard and
    left-right (lr) flipped orientations.

    Parameters
    ----------
    image : np.ndarray
        Processed brain image.
    voxel_sizes_mm : list[float]
        Voxel sizes in mm - [x, y, z].
    subject_dir : Path
        Subject directory to save nifti files.
    image_name : str
        Name to use as prefix for all saved files.

    Returns
    -------
    tuple[Path, Path]
        Returns (path of nifti image, path of lr-flipped nifti image)
    """
    image_path = subject_dir / f"{image_name}.nii.gz"
    flipped_image_path = subject_dir / f"{image_name}_lrflip.nii.gz"

    save_as_asr_nii(image, vox_sizes=voxel_sizes_mm, dest_path=image_path)

    flipped_image = np.flip(image, axis=2)
    save_as_asr_nii(
        flipped_image, vox_sizes=voxel_sizes_mm, dest_path=flipped_image_path
    )

    return image_path, flipped_image_path


def _process_subject(
    subject_row: pd.Series, config: PreprocConfig
) -> dict[str, Path]:
    """Process an individual subject's images.

    Parameters
    ----------
    subject_row : pd.Series
        Subject row from raw csv file.
    config : PreprocConfig
        Preprocessing config - contains settings for pre-processing steps.

    Returns
    -------
    dict[str, Path]
        Returns a dict of saved nifti paths:
        - image: path of processed brain image
        - mask: path of brain mask
        - flipped_image: path of lr-flipped processed brain image
        - flipped_mask: path of lr-flipped brain mask
    """

    image_path = Path(subject_row.source_filepath)
    subject_dir = _create_subject_dir(
        subject_row.subject_id, config.output_dir
    )

    image = load_any(image_path)
    vox_sizes_mm = [
        subject_row.resolution_z * 0.001,
        subject_row.resolution_y * 0.001,
        subject_row.resolution_x * 0.001,
    ]

    # n4 bias field correction
    image = correct_image_brightness(image, spacing=vox_sizes_mm)

    if ("mask_filepath" in subject_row) and pd.isnull(
        subject_row.mask_filepath
    ):
        mask_config = config.mask
        mask = create_mask(
            image,
            gauss_sigma=mask_config.gaussian_sigma,
            threshold_method=mask_config.threshold_method.value,
            closing_size=mask_config.closing_size,
            erode_size=mask_config.erode_size,
        )
    else:
        mask = load_any(subject_row.mask_filepath)

    # Crop image to mask bounds, and pad by n pixels
    image, mask = crop_to_mask(image, mask, padding=config.pad_pixels)

    # Make QC plots of the mask overlaid on the image
    plot_grid(
        image,
        overlay=mask,
        anat_space="ASR",
        section="frontal",
        overlay_is_mask=True,
        save_path=subject_dir / f"sub-{subject_row.subject_id}-QC-mask.png",
    )

    # Save image + mask, as well as flipped versions
    filename_prefix = image_path.stem.split(".")[0]
    image_path, flipped_image_path = _save_niftis(
        image, vox_sizes_mm[::-1], subject_dir, filename_prefix + "_processed"
    )
    mask_path, flipped_mask_path = _save_niftis(
        mask.astype("uint8"),
        vox_sizes_mm[::-1],
        subject_dir,
        filename_prefix + "_processed_mask",
    )
    return {
        "image": image_path,
        "mask": mask_path,
        "flipped_image": flipped_image_path,
        "flipped_mask": flipped_mask_path,
    }


def raw_to_ready(raw_csv: Path, config_file: Path) -> None:
    """Process nifti files in ASR orientation to create output images +
    masks ready for template creation.

    This assumes source_to_raw has already been run to downsample images,
    re-orient them to ASR and save them to the derivatives directory.

    raw_to_ready saves the following to each subject id's sub-dir
    inside the derivatives directory:
    - ..._processed.nii.gz : the processed brain image
    - ..._processed_mask.nii.gz : the mask of the brain image
    - ..._processed_lrflip.nii.gz : the lr-flipped processed brain image
    - ..._processed_mask_lrflip.nii.gz : the lr-flipped mask of the brain image
    - ..-QC-mask.png: a plot showing the mask overlaid on the brain image

    At the top level of the derivatives dir, two text files are produced:
    - all_processed_brain_paths.txt : paths of processed images (including
    flipped) for all subject ids
    - all_processed_mask_paths.txt : paths of masks (including flipped)
    for all subject ids

    Parameters
    ----------
    raw_csv : Path
        Raw csv file path. One row per sample, each with a
        unique 'subject_id' - this is created via `source_to_raw`.
    config_file : Path
        Config json file path. Contains settings for pre-processing steps.
    """

    validate_input_csv(raw_csv)
    input_df = pd.read_csv(raw_csv)

    with open(config_file) as f:
        config_yaml = yaml.safe_load(f)

    config = PreprocConfig.model_validate(config_yaml)
    derivatives_dir = config.output_dir / DERIVATIVES_DIR_NAME
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    mask_paths = []
    for _, row in input_df.iterrows():

        if ("use" in row) and (row.use is False):
            continue

        paths_dict = _process_subject(row, config)
        image_paths.extend([paths_dict["image"], paths_dict["flipped_image"]])
        mask_paths.extend([paths_dict["mask"], paths_dict["flipped_mask"]])

    np.savetxt(
        derivatives_dir / "all_processed_brain_paths.txt",
        image_paths,
        fmt="%s",
    )
    np.savetxt(
        derivatives_dir / "all_processed_mask_paths.txt",
        mask_paths,
        fmt="%s",
    )

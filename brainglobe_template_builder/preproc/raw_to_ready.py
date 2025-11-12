from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image.load import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.io import get_unique_folder_in_dir
from brainglobe_template_builder.plots import plot_grid
from brainglobe_template_builder.preproc.cropping import crop_to_mask
from brainglobe_template_builder.preproc.masking import create_mask
from brainglobe_template_builder.preproc.preproc_config import PreprocConfig


def _get_sample_image_path(subject_id: str, derivatives_dir: Path) -> Path:
    subject_dir = get_unique_folder_in_dir(derivatives_dir, subject_id)
    image_paths = list(
        subject_dir.glob(f"sub-{subject_id}*_origin-asr.nii.gz")
    )

    if len(image_paths) != 1:
        raise ValueError(
            f"Expected one image file for {subject_id}: found {image_paths}"
        )

    return image_paths[0]


def _save_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    voxel_sizes_mm: list[float],
    output_dir: Path,
    image_name: str,
) -> dict[str, Path]:
    """Save image and mask as nifti files - in standard and
    x-flipped orientations.

    Parameters
    ----------
    image : np.ndarray
        Processed brain image.
    mask : np.ndarray
        Mask of processed brain image.
    voxel_sizes_mm : list[float]
        Voxel sizes in mm - [x, y, z].
    output_dir : Path
        Output directory to save nifti files.
    image_name : str
        Name to use as prefix for all saved files.

    Returns
    -------
    dict[str, Path]
        Returns a dict of saved nifti paths:
        - image: path of processed brain image
        - mask: path of brain mask
        - flipped_image: path of x-flipped processed brain image
        - flipped_mask: path of x-flipped brain mask
    """

    image_path = output_dir / f"{image_name}_processed.nii.gz"
    mask_path = output_dir / f"{image_name}_processed_mask.nii.gz"
    flipped_image_path = output_dir / f"{image_name}_processed_xflip.nii.gz"
    flipped_mask_path = (
        output_dir / f"{image_name}_processed_mask_xflip.nii.gz"
    )

    save_as_asr_nii(image, vox_sizes=voxel_sizes_mm, dest_path=image_path)
    save_as_asr_nii(
        mask.astype(np.uint8), vox_sizes=voxel_sizes_mm, dest_path=mask_path
    )

    flipped_image = np.flip(image, axis=2)
    flipped_mask = np.flip(mask, axis=2)
    save_as_asr_nii(
        flipped_image, vox_sizes=voxel_sizes_mm, dest_path=flipped_image_path
    )
    save_as_asr_nii(
        flipped_mask.astype(np.uint8),
        vox_sizes=voxel_sizes_mm,
        dest_path=flipped_mask_path,
    )

    return {
        "image": image_path,
        "mask": mask_path,
        "flipped_image": flipped_image_path,
        "flipped_mask": flipped_mask_path,
    }


def _process_subject(
    subject_id: str, config: PreprocConfig
) -> dict[str, Path]:
    """Process an individual subject's images.

    Parameters
    ----------
    subject_id : str
        Unique subject id.
    config : PreprocConfig
        Preprocessing config - contains settings for pre-processing steps.

    Returns
    -------
    dict[str, Path]
        Returns a dict of saved nifti paths:
        - image: path of processed brain image
        - mask: path of brain mask
        - flipped_image: path of x-flipped processed brain image
        - flipped_mask: path of x-flipped brain mask
    """

    image_path = _get_sample_image_path(subject_id, config.derivatives_dir)
    output_dir = image_path.parent
    image = load_any(image_path)
    vox_sizes_mm = [
        config.resolution_x * 0.001,
        config.resolution_y * 0.001,
        config.resolution_z * 0.001,
    ]

    # TODO - denoising
    # TODO - n4 bias field correction

    mask_config = config.mask
    mask = create_mask(
        image,
        gauss_sigma=mask_config.gaussian_sigma,
        threshold_method=mask_config.threshold_method.value,
        closing_size=mask_config.closing_size,
        erode_size=mask_config.erode_size,
    )

    # Crop image to mask bounds, and pad by n pixels
    image, mask = crop_to_mask(image, mask, padding=config.pad_pixels)

    # Make QC plots of the mask overlaid on the image
    plot_grid(
        image,
        overlay=mask,
        anat_space="ASR",
        section="frontal",
        overlay_is_mask=True,
        save_path=output_dir / f"sub-{subject_id}-QC-mask.png",
    )

    # Save image + mask, as well as flipped versions
    return _save_image_and_mask(
        image, mask, vox_sizes_mm, output_dir, image_path.stem
    )


def raw_to_ready(input_csv: Path, config_file: Path) -> None:
    """Process nifti files in ASR orientation to create output images +
    masks ready for template creation.

    This assumes source_to_raw has already been run to downsample images,
    re-orient them to ASR and save them to the derivatives directory.

    raw_to_ready saves the following to each subject id's sub-dir
    inside the derivatives directory:
    - ..._processed.nii.gz : the processed brain image
    - ..._processed_mask.nii.gz : the mask of the brain image
    - ..._processed_xflip.nii.gz : the x-flipped processed brain image
    - ..._processed_mask_xflip.nii.gz : the x-flipped mask of the brain image
    - ..-QC-mask.png: a plot showing the mask overlaid on the brain image

    At the top level of the derivatives dir, two text files are produced:
    - all_processed_brain_paths.txt : paths of processed images (including
    flipped) for all subject ids
    - all_processed_mask_paths.txt : paths of masks (including flipped)
    for all subject ids

    Parameters
    ----------
    input_csv : Path
        Input csv file path. One row per sample, each with a
        unique 'subject_id'.
    config_file : Path
        Config json file path. Contains settings for pre-processing steps.
    """

    input_df = pd.read_csv(input_csv)
    # TODO - Validate input csv

    config_json = config_file.read_text()
    config = PreprocConfig.model_validate_json(config_json)

    image_paths = []
    mask_paths = []
    for subject_id in input_df["subject_id"]:

        if ("use" in input_df) and (input_df["use"] is False):
            continue

        paths_dict = _process_subject(subject_id, config)
        image_paths.extend([paths_dict["image"], paths_dict["flipped_image"]])
        mask_paths.extend([paths_dict["mask"], paths_dict["flipped_mask"]])

    np.savetxt(
        config.derivatives_dir / "all_processed_brain_paths.txt",
        image_paths,
        fmt="%s",
    )
    np.savetxt(
        config.derivatives_dir / "all_processed_mask_paths.txt",
        mask_paths,
        fmt="%s",
    )

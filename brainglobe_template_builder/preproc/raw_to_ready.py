from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image.load import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.io import get_unique_folder_in_dir
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


def process_sample(
    subject_id: str, config: PreprocConfig
) -> tuple[Path, Path]:

    image_path = _get_sample_image_path(subject_id, config.derivatives_dir)
    image = load_any(image_path)

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

    # TODO - xflip

    output_dir = image_path.parent
    image_save_path = output_dir / f"{image_path.stem}_processed.nii.gz"
    mask_save_path = output_dir / f"{image_path.stem}_processed_mask.nii.gz"
    vox_sizes_mm = [
        config.resolution_x * 0.001,
        config.resolution_y * 0.001,
        config.resolution_z * 0.001,
    ]

    save_as_asr_nii(image, vox_sizes=vox_sizes_mm, dest_path=image_save_path)
    save_as_asr_nii(
        mask.astype(np.uint8), vox_sizes=vox_sizes_mm, dest_path=mask_save_path
    )

    return image_save_path, mask_save_path


def raw_to_ready(input_csv: Path, config_file: Path) -> None:

    input_df = pd.read_csv(input_csv)
    # TODO - Validate input csv

    config_json = config_file.read_text()
    config = PreprocConfig.model_validate_json(config_json)

    for subject_id in input_df["subject_id"]:

        if ("use" in input_df) and (input_df["use"] is False):
            continue

        process_sample(subject_id, config)

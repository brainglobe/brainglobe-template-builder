import argparse
from pathlib import Path

import ants
import numpy as np
import pandas as pd
from brainglobe_utils.IO.image import load_any, save_any
from dask import array as da
from loguru import logger

from brainglobe_template_builder.io import (
    file_path_with_suffix,
    save_as_asr_nii,
)
from brainglobe_template_builder.preproc.masking import create_mask
from brainglobe_template_builder.preproc.transform_utils import (
    downsample_anisotropic_image_stack,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download source image")
    parser.add_argument(
        "--source_data_root",
        type=str,
        help="Path to the source data folder. The source data should contain"
        "a subfolder per subject, with picture files within it",
        required=True,
    )
    parser.add_argument(
        "--template_building_root",
        type=str,
        help="Path to the template-building root folder.Results will be "
        "written to the rawdata folder.",
        required=True,
    )
    parser.add_argument(
        "--target_isotropic_resolution",
        type=int,
        help="Target isotropic resolution",
        required=True,
    )

    parser.add_argument(
        "--data_catalog",
        type=str,
        help="The full path to the data catalog file",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset used to process",
        required=True,
    )

    args = parser.parse_args()

    source_data = Path(args.source_data_root)
    template_building_root = Path(args.template_building_root)
    target_isotropic_resolution = int(args.target_isotropic_resolution)

    in_plane_resolution = 0.55
    out_of_plane_resolution = 1

    in_plane_factor = int(
        np.ceil(target_isotropic_resolution / in_plane_resolution)
    )
    axial_factor = int(
        np.ceil(target_isotropic_resolution / out_of_plane_resolution)
    )

    template_raw_data = template_building_root / "rawdata"
    template_raw_data.mkdir(exist_ok=True, parents=True)

    # Load the data catalog from argument
    data_catalog_path = Path(args.data_catalog)
    data_catalog = pd.read_csv(data_catalog_path)
    dataset = args.dataset

    # Specified the dataset to process
    dataset_catalog = data_catalog[data_catalog["dataset"] == dataset]
    logger.debug(f"Loaded {dataset} dataset catalog from {data_catalog_path}.")

    # Check if there are right wing discs is in the dataset
    right_wingdisc_catalog = dataset_catalog[dataset_catalog["is_left"] == "n"]
    if right_wingdisc_catalog.empty:
        logger.info(f"No right wing discs found in {dataset} dataset.")

    for sample_folder in source_data.iterdir():
        # Load the images and specify the filename of processed images
        logger.info(f"Downsampling {sample_folder}...")
        sample_id = str(sample_folder.name).split("_")[0].lower()
        channel = "membrane"
        downsampled_filename = (
            f"{sample_id}_res-{target_isotropic_resolution}"
            f"um_channel-{channel}.tif"
        )
        assert Path(sample_folder).exists(), f"{sample_folder} not found"
        original_file_path = Path(sample_folder) / f"{sample_folder.name}.tif"
        assert Path(
            original_file_path
        ).exists(), f"Filepath {original_file_path} not found"
        image_array = load_any(original_file_path)

        # Do mirroring if the sample is right wing disc
        if (
            str(sample_folder.name)
            in right_wingdisc_catalog["filename"].astype(str).tolist()
        ):
            image_array = np.flip(image_array, axis=2)
            logger.info(f"Mirrored {sample_folder.name}.")

        # Downsample the image array
        image_dask = da.from_array(image_array, chunks={0: 1, 1: -1, 2: -1})
        down_sampled_image = downsample_anisotropic_image_stack(
            image_dask, in_plane_factor, axial_factor
        )
        down_sampled_image = down_sampled_image.astype(np.uint16)

        # Save the downsampled image as tif
        saving_folder = (
            template_raw_data
            / f"{source_data.name}"
            / downsampled_filename.split(".")[0]
        )
        saving_folder.mkdir(exist_ok=True, parents=True)
        assert Path(
            saving_folder
        ).exists(), f"Filepath {saving_folder} not found"
        saving_path = saving_folder / downsampled_filename
        save_any(down_sampled_image, saving_path)
        logger.info(
            f"{sample_folder} downsampled, saved as {downsampled_filename}"
        )

        # Save the downsampled image as nifti
        nii_path = file_path_with_suffix(
            saving_path, "_downsampled", new_ext=".nii.gz"
        )
        vox_sizes = [
            target_isotropic_resolution,
        ] * 3
        save_as_asr_nii(down_sampled_image, vox_sizes, nii_path)
        logger.info(f"Saved downsampled image as {nii_path.name}.")

        # Generate the wingdisc mask
        image_ants = ants.image_read(nii_path.as_posix())
        mask_data = create_mask(
            image_ants.numpy(),
            gauss_sigma=10,
            threshold_method="triangle",
            closing_size=5,
        )
        mask_path = file_path_with_suffix(nii_path, "_mask")
        mask = image_ants.new_image_like(mask_data.astype(np.uint8))
        ants.image_write(mask, mask_path.as_posix())
        logger.debug(
            f"Generated brain mask with shape: {mask.shape} "
            f"and saved as {mask_path.name}."
        )

        # Plot the mask over the image to check
        mask_plot_path = (
            saving_folder / f"{sample_id}_downsampled_mask-overlay.png"
        )
        ants.plot(
            image_ants,
            mask,
            overlay_alpha=0.5,
            axis=1,
            title="Wingdisc mask over image",
            filename=mask_plot_path.as_posix(),
        )
        logger.debug("Plotted overlay to visually check mask.")

        # Write a text file to record the file path to the downsampled image and mask created
        with open("brain_path.txt", "a") as f:
            f.write(f"{nii_path} + \n")
        with open("mask_path.txt", "a") as f:
            f.write(f"{mask_path} + \n")
        logger.debug(
            "Recorded the file path to the brain_path.txt and mask_path.txt created."
        )

        # Write a text file to record the file path to the downsampled image and mask created
        nii_path_str = str(nii_path)
        downsampled_txt_file_name = "brain_paths.txt"
        downsampled_txt_file_path = (
            template_raw_data
            / f"{source_data.name}"
            / downsampled_txt_file_name
        )
        mask_path_str = str(mask_path)
        mask_txt_file_name = "mask_paths.txt"
        mask_txt_file_path = (
            template_raw_data / f"{source_data.name}" / mask_txt_file_name
        )
        # Read the file(if exist first) to check if the path already exists
        try:
            with open(downsampled_txt_file_path, "r", encoding="utf-8") as f:
                lines_downsampled = (
                    f.read().splitlines()
                )  # Read lines without newlines
            with open(mask_txt_file_path, "r", encoding="utf-8") as f:
                lines_mask = f.read().splitlines()
        except FileNotFoundError:
            lines_downsampled = []
            lines_mask = []  # If file doesn't exist, start with an empty list
        if nii_path_str not in lines_downsampled:
            with open(downsampled_txt_file_path, "a", encoding="utf-8") as f:
                f.write(f"{nii_path_str}\n")
                logger.debug(
                    f"Recorded {nii_path_str} in {downsampled_txt_file_path}."
                )
        if mask_path_str not in lines_mask:
            with open(mask_txt_file_path, "a", encoding="utf-8") as f:
                f.write(f"{mask_path_str}\n")
                logger.debug(
                    f"Recorded {mask_path_str} in {mask_txt_file_path}."
                )

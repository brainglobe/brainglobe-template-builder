import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image import load_any, save_any
from loguru import logger
from skimage import transform

from brainglobe_template_builder.plots import plot_grid, plot_orthographic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample source images")
    parser.add_argument(
        "--template_building_root",
        type=str,
        help="Path to the template-building root folder. Results will "
        "be written to the rawdata subfolder.",
        required=True,
    )
    parser.add_argument(
        "--target_isotropic_resolution",
        type=int,
        help="Target isotropic resolution",
        required=True,
    )

    args = parser.parse_args()

    atlas_forge_molerat_path = Path(
        "/media/ceph/neuroinformatics/neuroinformatics/atlas-forge/MoleRat/"
    )
    source_data = atlas_forge_molerat_path / "new_d06"
    source_info_file = atlas_forge_molerat_path / "molerat_brains_info_PM.csv"

    template_building_root = Path(args.template_building_root)
    target_isotropic_resolution = int(args.target_isotropic_resolution)

    in_plane_resolution = 20
    out_of_plane_resolution = 20
    in_plane_factor = round(target_isotropic_resolution / in_plane_resolution)
    axial_factor = round(target_isotropic_resolution / out_of_plane_resolution)

    template_raw_data = template_building_root / "rawdata"
    template_raw_data.mkdir(exist_ok=True, parents=True)

    source_info = pd.read_csv(source_info_file, skiprows=1, header=0)
    logger.info(f"Loaded source info with {len(source_info)} entries.")
    counter = 0
    for _, sample in source_info.iterrows():
        if sample["comments"] != "restitched":
            continue
        original_slice_direction = sample["original_slice_direction"]
        if original_slice_direction == "horizontal":
            source_file = (
                source_data / "HorizontallyImaged" / sample["filename"]
            )
        elif original_slice_direction == "sagittal":
            source_file = source_data / "SagitallyImaged" / sample["filename"]
        else:
            raise ValueError(
                f"Unexpected slice direction {original_slice_direction}"
            )
        assert source_file.exists(), f"File {source_file} not found"

        subject_id = sample["subject_id"]
        hemisphere = sample["hemisphere"]
        subject_folder = (
            template_raw_data / f"sub-{subject_id}_hemi-{hemisphere}"
        )
        subject_folder.mkdir(exist_ok=True)
        rawdata_filename = (
            f"sub-{subject_id}_"
            f"hemi-{hemisphere}_"
            f"res-{target_isotropic_resolution}um.tif"
        )

        # we can't use our usual transform utils function here,
        # because it's not a dask array,
        # and we additionally need to mirror and reorient the stack to ASR
        assert (
            sample["image_orientation"] == "RPI"
        ), "Image orientation is not RPI"
        stack = load_any(str(source_file))

        if sample["looks_like_right_hemisphere"] == "no":
            logger.info(
                f"Flipping left hemisphere to right for {str(source_file)}"
            )
            stack = np.flip(stack, axis=0)
        # Find the last few slices of stack that contain all zeros
        zero_slices = 0
        for i in range(stack.shape[0] - 1, -1, -1):
            if np.all(stack[i] == 0):
                zero_slices += 1
            else:
                break
        if zero_slices:
            logger.info(f"Last zero slices: {zero_slices}")
            stack = stack[: stack.shape[0] - zero_slices, :, :]

        # mirror BEFORE downsampling
        # (because otherwise midline gets blurred by zeros!)
        stack = np.concatenate((stack, np.flip(stack, axis=0)), axis=0)
        if target_isotropic_resolution != 20:
            downsampled = transform.downscale_local_mean(
                stack, (axial_factor, in_plane_factor, in_plane_factor)
            )
        else:
            logger.info("No downsampling needed!")
            downsampled = stack
        original_space = AnatomicalSpace("RPI")
        downsampled = original_space.map_stack_to("ASR", downsampled)
        save_any(downsampled, subject_folder / rawdata_filename)

        plots_folder = (
            Path.home() / "dev/brainglobe-template-builder/test-images/"
        )
        plot_grid(
            downsampled,
            save_path=plots_folder / f"grid-{rawdata_filename}.png",
        )
        plot_orthographic(
            downsampled,
            save_path=plots_folder / f"ortho-{rawdata_filename}.png",
        )
        logger.info(f"{rawdata_filename} downsampled.")

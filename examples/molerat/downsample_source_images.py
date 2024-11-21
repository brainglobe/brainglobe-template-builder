import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image import load_any, save_any
from loguru import logger
from skimage import transform

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
        "/media/ceph-neuroinformatics/neuroinformatics/atlas-forge/MoleRat/"
    )
    source_data = (
        atlas_forge_molerat_path
        / "Mole-rat brain atlas (Fukomys anselli)_MalkemperLab"
    )
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
        if sample["comments"] != "good":
            logger.info(f"Skipping {source_file.name} for now")
            continue

        subject_id = sample["subject_id"]
        hemisphere = sample["hemisphere"]
        subject_folder = (
            template_raw_data / f"sub-{subject_id}_hemi-{hemisphere}"
        )
        subject_folder.mkdir(exist_ok=True)
        rawdata_filename = (
            f"sub-{subject_id}_"
            f"hemi-{hemisphere}_"
            f"res-{target_isotropic_resolution}.tif"
        )

        # we can't use our usual transform utils function here,
        # because it's not a dask array,
        # and we additionally need to mirror and reorient the stack to ASR
        assert sample[
            "looks_like_right_hemisphere"
        ], f"TODO: flip sample {source_file}"
        assert (
            sample["image_orientation"] == "RPI"
        ), "Image orientation is not RPI"
        stack = load_any(str(source_file))
        downsampled = transform.downscale_local_mean(
            stack, (axial_factor, in_plane_factor, in_plane_factor)
        )
        downsampled = np.concatenate(
            (downsampled, np.flip(downsampled, axis=0)), axis=0
        )
        original_space = AnatomicalSpace("RPI")
        downsampled = original_space.map_stack_to("ASR", downsampled)
        save_any(downsampled, subject_folder / rawdata_filename)

        logger.info(f"{rawdata_filename} downsampled.")

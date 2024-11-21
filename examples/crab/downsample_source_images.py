import argparse
from pathlib import Path

from loguru import logger

from brainglobe_template_builder.preproc.transform_utils import downsample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample source images")
    parser.add_argument(
        "--source_data",
        type=str,
        help="Path to the source data folder. The source data should "
        "contain a subfolder per subject, and tiffs within it.",
        required=True,
    )
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

    source_data = Path(args.source_data)
    template_building_root = Path(args.template_building_root)
    target_isotropic_resolution = int(args.target_isotropic_resolution)

    in_plane_resolution = 0.43
    out_of_plane_resolution = 5
    in_plane_factor = round(target_isotropic_resolution / in_plane_resolution)
    axial_factor = round(target_isotropic_resolution / out_of_plane_resolution)

    template_raw_data = template_building_root / "rawdata"
    template_raw_data.mkdir(exist_ok=True, parents=True)

    for sample_folder in source_data.iterdir():
        if (
            not str(sample_folder)[-6:-1] == "Crab_"
        ):  # find folders like Crab_3
            logger.info(f"{str(sample_folder)[-6:-1]} is not a crab folder")
            continue
        logger.info(f"Downsampling {sample_folder.name}...")
        sample_id = str(sample_folder).split("_")[-1].lower()
        stain = "Nuclear"
        sample_filename = (
            f"sub-{sample_id}_res-{target_isotropic_resolution}"
            f"um_stain-{stain}.tif"
        )
        downsample(
            sample_folder / "Nuclear/",
            template_raw_data / sample_filename,
            in_plane_factor,
            axial_factor,
        )
        logger.info(f"{sample_filename} downsampled.")

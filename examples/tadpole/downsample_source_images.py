import argparse
from pathlib import Path

from brainglobe_utils.IO.image import read_z_stack, save_any
from loguru import logger

from brainglobe_template_builder.preproc.transform_utils import (
    downsample_anisotropic_image_stack,
)


def downsample_tadpole(folder: Path) -> None:
    logger.info(f"Downsampling {folder.name}...")
    stack = read_z_stack(str(folder))

    in_plane_resolution = 1
    out_of_plane_resolution = 3
    in_plane_factor = int(target_isotropic_resolution / in_plane_resolution)
    axial_factor = int(target_isotropic_resolution / out_of_plane_resolution)
    downsampled = downsample_anisotropic_image_stack(
        stack, in_plane_factor=in_plane_factor, axial_factor=axial_factor
    )

    sample_id = str(folder).split("_")[1].lower()
    channel = "blue" if str(folder).split("_")[2] == "488" else "orange"
    sample_filename = (
        f"sub-{sample_id}_res-{target_isotropic_resolution}"
        f"um_channel-{channel}.tif"
    )
    save_any(downsampled, template_raw_data / sample_filename)

    logger.info(f"{sample_filename} downsampled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample source images")
    parser.add_argument(
        "--source_data",
        type=str,
        help="Path to the source data folder. The source data should " \
        "contain a subfolder per subject, and tiffs within it.",
        required=True,
    )
    parser.add_argument(
        "--template_building_root",
        type=str,
        help="Path to the template-building root folder. Results will " \
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

    for subfolder_name in [
        "rawdata",
        "logs",
        "derivatives",
        "scripts",
        "templates",
    ]:
        subfolder = template_building_root / subfolder_name
        subfolder.mkdir(exist_ok=True, parents=True)

    template_raw_data = template_building_root / "rawdata"
    for folder in source_data.iterdir():
        downsample_tadpole(folder)

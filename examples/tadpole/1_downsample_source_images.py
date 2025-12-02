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

    in_plane_resolution = 1
    out_of_plane_resolution = 3
    in_plane_factor = int(target_isotropic_resolution / in_plane_resolution)
    axial_factor = int(target_isotropic_resolution / out_of_plane_resolution)

    template_raw_data = template_building_root / "rawdata"
    template_raw_data.mkdir(exist_ok=True, parents=True)

    for sample_folder in source_data.iterdir():
        logger.info(f"Downsampling {sample_folder.name}...")
        sample_id = str(sample_folder).split("_")[1].lower()
        channel = (
            "blue" if str(sample_folder).split("_")[2] == "488" else "orange"
        )
        sample_filename = (
            f"sub-{sample_id}_res-{target_isotropic_resolution}"
            f"um_channel-{channel}.tif"
        )
        downsample(
            sample_folder,
            template_raw_data / sample_filename,
            [1 / axial_factor, 1 / in_plane_factor, 1 / in_plane_factor],
        )
        logger.info(f"{sample_filename} downsampled.")

import argparse
from pathlib import Path
import pandas as pd

from brainglobe_utils.IO.image import save_any, load_any
from dask import array as da
from brainglobe_template_builder.preproc.load_wingdisc import load_images
from loguru import logger

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
    '''
    parser.add_argument(
        "--data_catalog",
        type=str,
        help="The full path to the data catalog file",
        required=True,
    )
    '''
    '''
    parser.add_argument(
        "--binning",
        type=str,
        help="Whether the required data should be binned first. Options are 'y' for yes and 'n' for no",
        required=True,
    )
    parser.add_argument(
        "--average",
        type=int,
        help="Whether the required data should be averaged first. Options are 'y' for yes and 'n' for no",
        required=True,
    )
    parser.add_argument(
        "--camera_exposure_time",
        type=int,
        help="The camera exposure time specific for the data required",
        required=True,
    )
    '''

    args = parser.parse_args()

    source_data = Path(args.source_data_root)
    template_building_root = Path(args.template_building_root)
    target_isotropic_resolution = int(args.target_isotropic_resolution)
    #data_catalog_path = Path(args.data_catalog)

    in_plane_resolution = 0.55
    out_of_plane_resolution = 1

    in_plane_factor = int(target_isotropic_resolution / in_plane_resolution)
    axial_factor = int(target_isotropic_resolution / out_of_plane_resolution)

    template_raw_data = template_building_root / "rawdata"
    template_raw_data.mkdir(exist_ok=True, parents=True)

    #data_catalog = pd.read_csv(data_catalog_path)

    for sample_folder in source_data.iterdir():
        logger.info(f"Downsampling {sample_folder}...")
        sample_id = str(sample_folder.name).split("_")[0].lower()
        channel = "membrane"
        downsampled_filename = (
            f"sub-{sample_id}_res-{target_isotropic_resolution}"
            f"um_channel-{channel}.tif"
        )
        assert Path(sample_folder).exists(), f"{sample_folder} not found"
        original_file_path = (
            Path(sample_folder)
            / f"{sample_folder.name}.tif"
        )
        assert Path(
            original_file_path
        ).exists(), f"Filepath {original_file_path} not found"

        image_array = load_any(
            original_file_path
        )
        image_dask = da.from_array(
            image_array, chunks={0: 1, 1: -1, 2: -1}
        )
        down_sampled_image = downsample_anisotropic_image_stack(
            image_dask, in_plane_factor, axial_factor
        )

        saving_path = template_raw_data / f'{source_data.name}'/ downsampled_filename.split('.')[0]
        saving_path.mkdir(exist_ok=True, parents=True)
        assert Path(
            saving_path
        ).exists(), f"Filepath {saving_path} not found"
        save_any(down_sampled_image, saving_path/downsampled_filename)
        logger.info(f"{sample_folder} downsampled, saved as {downsampled_filename}")


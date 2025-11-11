"""
Prepare low-resolution Molerat images for template construction
================================================================
The following operations are performed on the lowest-resolution images to
prepare them for template construction:
- Perform N4 Bias field correction using ANTs
- Generate brain mask based on N4-corrected image
- Save all resulting images as nifti files to be used for template construction
"""

# %%
# Imports
# -------
import os
from datetime import date
from pathlib import Path

import ants
import numpy as np
from brainglobe_utils.IO.image.save import save_as_asr_nii
from loguru import logger

from brainglobe_template_builder.io import file_path_with_suffix, load_tiff
from brainglobe_template_builder.preproc.masking import create_mask
from brainglobe_template_builder.preproc.splitting import (
    get_right_and_left_slices,
    save_array_dict_to_nii,
)

# %%
# Setup
# -----
# Define some global variables, including paths to input and output directories
# and set up logging.

# Define voxel size(in microns) of the lowest resolution image
lowres = 40
# String to identify the resolution in filenames
res_str = f"res-{lowres}um"
# Define voxel sizes in mm (for Nifti saving)
vox_sizes = [lowres * 1e-3] * 3  # in mm

# Prepare directory structure
atlas_dir = Path("/media/ceph/neuroinformatics/neuroinformatics/atlas-forge")
species_id = "MoleRat"
species_dir = atlas_dir / species_id
raw_dir = species_dir / "rawdata"
assert raw_dir.exists(), f"Could not find rawdata directory {raw_dir}."
deriv_dir = species_dir / "derivatives"
assert deriv_dir.exists(), f"Could not find derivatives directory {deriv_dir}."

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(species_dir / "logs" / f"{today}_{current_script_name}.log")


# %%
# Run the pipeline for each subject
# ---------------------------------

# Create a dictionary to store the paths to the use4template directories
# per subject. These will contain all necessary images for template building.
use4template_dirs = {}

for raw_subj_dir in raw_dir.iterdir():

    file_prefix = f"{raw_subj_dir.name}_{res_str}"
    deriv_subj_dir = deriv_dir / raw_subj_dir.name
    deriv_subj_dir.mkdir(exist_ok=True)
    raw_tiff_path = raw_subj_dir / f"{file_prefix}.tif"
    assert raw_tiff_path.exists()

    logger.info(f"Starting to process {file_prefix}...")
    logger.info(f"Will save outputs to {deriv_subj_dir}/")

    # Load the image (already in ASR)
    image = load_tiff(raw_tiff_path)
    logger.debug(
        f"Loaded image {raw_tiff_path.name} with shape: {image.shape}."
    )

    # Save the image as nifti
    nii_path = file_path_with_suffix(
        deriv_subj_dir / f"{file_prefix}.tif", "_orig-asr", new_ext=".nii.gz"
    )
    save_as_asr_nii(image, vox_sizes, nii_path)
    logger.debug(f"Saved reoriented image as {nii_path.name}.")

    # Bias field correction (to homogenise intensities)
    image_ants = ants.image_read(nii_path.as_posix())
    image_n4 = ants.n4_bias_field_correction(image_ants)
    image_n4_path = file_path_with_suffix(nii_path, "_N4")
    ants.image_write(image_n4, image_n4_path.as_posix())
    logger.debug(
        f"Created N4 bias field corrected image as {image_n4_path.name}."
    )

    # Generate a brain mask based on the N4-corrected image
    mask_data = create_mask(
        image_n4.numpy(),
        gauss_sigma=3,
        threshold_method="triangle",
        closing_size=5,
    )
    mask_path = file_path_with_suffix(nii_path, "_N4_mask")
    mask = image_n4.new_image_like(mask_data.astype(np.uint8))
    ants.image_write(mask, mask_path.as_posix())
    logger.debug(
        f"Generated brain mask with shape: {mask.shape} "
        f"and saved as {mask_path.name}."
    )

    # Plot the mask over the image to check
    mask_plot_path = (
        deriv_subj_dir / f"{file_prefix}_orig-asr_N4_mask-overlay.png"
    )
    ants.plot(
        image_n4,
        mask,
        overlay_alpha=0.5,
        axis=1,
        title="Brain mask over image",
        filename=mask_plot_path.as_posix(),
    )
    logger.debug("Plotted overlay to visually check mask.")

    output_prefix = file_path_with_suffix(nii_path, "_N4_aligned_", new_ext="")
    # Generate arrays for template construction and save as niftis
    use4template_dir = Path(output_prefix.as_posix() + "padded_use4template")
    # if it exists, delete existing files in it
    if use4template_dir.exists():
        logger.warning(f"Removing existing files in {use4template_dir}.")
        for file in use4template_dir.glob("*"):
            file.unlink()
    use4template_dir.mkdir(exist_ok=True)

    image_n4 = image_n4.numpy()
    mask = mask.numpy()
    right_hemi_slices, _ = get_right_and_left_slices(image_n4)

    array_dict = {
        f"{use4template_dir}/{file_prefix}_sym-brain": np.pad(
            image_n4, pad_width=2, mode="constant"
        ),
        f"{use4template_dir}/{file_prefix}_sym-mask": np.pad(
            mask, pad_width=2, mode="constant"
        ),
        f"{use4template_dir}/{file_prefix}_right-hemi-brain": np.pad(
            image_n4[right_hemi_slices], pad_width=2, mode="constant"
        ),
        f"{use4template_dir}/{file_prefix}_right-hemi-mask": np.pad(
            mask[right_hemi_slices], pad_width=2, mode="constant"
        ),
    }
    save_array_dict_to_nii(array_dict, use4template_dir, vox_sizes)
    use4template_dirs[file_prefix] = use4template_dir
    logger.info(
        f"Saved images for template construction in {use4template_dir}."
    )
    logger.info(f"Finished processing {file_prefix}.")

# %%
# Generate lists of file paths for template construction
# -----------------------------------------------------
# Use the paths to the use4template directories to generate lists of file paths
# for the template construction pipeline. Three kinds of template will be
# generated, and each needs the corresponding brain image and mask files:
# 1. All asym* files for subjects where hemi=both. These will be used to
#    generate an asymmetric brain template.
# 2. All right-sym* files for subjects where use_right is True, and
#    all left-sym* files for subjects where use_left is True.
#    These will be used to generate a symmetric brain template.
# 3. All right-hemi* files for subjects where use_right is True,
#    and all left-hemi-xflip* files for subjects where use_left is True.
#    These will be used to generate a symmetric hemisphere template.

filepath_lists: dict[str, list] = {
    "sym-brain": [],
    "sym-mask": [],
    "hemi-brain": [],
    "hemi-mask": [],
}

for subject, use4template_dir in use4template_dirs.items():
    for label in ["brain", "mask"]:
        filepath_lists[f"sym-{label}"].append(
            use4template_dir / f"{subject}_sym-{label}.nii.gz"
        )
        filepath_lists[f"hemi-{label}"].append(
            use4template_dir / f"{subject}_right-hemi-{label}.nii.gz"
        )

# %%
# Save the file paths to text files, each in a separate directory

for key, paths in filepath_lists.items():
    kind, label = key.split("-")  # e.g. "asym" and "brain"
    n_images = len(paths)
    template_name = f"template_{kind}_{res_str}_n-{n_images}"
    template_dir = species_dir / "templates" / template_name
    template_dir.mkdir(exist_ok=True)
    np.savetxt(template_dir / f"{label}_paths.txt", paths, fmt="%s")

# %%

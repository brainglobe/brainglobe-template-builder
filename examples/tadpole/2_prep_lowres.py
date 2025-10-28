"""
Prepare low-resolution tadpole images for template construction
================================================================
The ``brainglobe-template-builder`` Preprocess widget was used to perform
the following operations on each subject's low-resolution image, ahead of
running this script:
- Load each image from tiff, re-orient to ASR and save as nifti
- Generate a brain mask automatically and modify it manually, if necessary
- Find the mid-sagittal plane and align the image and mask accordingly
- Save the aligned image, brain mask, and halves mask as nifti files

This script will:
- Perform N4 Bias field correction on the aligned image using ANTs
- Split the image and mask into hemispheres and reflect each hemisphere
- Generate symmetric brains using either the left or right hemisphere
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
import pandas as pd
from loguru import logger
from tqdm import tqdm

from brainglobe_template_builder.io import (
    file_path_with_suffix,
)
from brainglobe_template_builder.preproc.splitting import (
    generate_arrays_4template,
    save_array_dict_to_nii,
)

# %%
# Setup
# -----
# Define some global variables, including paths to input and output directories
# and set up logging.

# Define voxel size(in microns) of the lowest resolution image
lowres = 12
# String to identify the resolution in filenames
res_str = f"res-{lowres}um"
# Define voxel sizes in mm (for Nifti saving)
vox_sizes = [lowres * 1e-3] * 3  # in mm

# Masking parameters
mask_params = {
    "gauss_sigma": 2,
    "threshold_method": "triangle",
    "closing_size": 3,
}

# Prepare directory structure
atlas_forge_dir = Path("/media/ceph-niu/neuroinformatics/atlas-forge")
project_dir = (
    atlas_forge_dir / "Tadpole" / "tadpole-template-starter" / "old-tadpole"
)
raw_dir = project_dir / "rawdata"
deriv_dir = project_dir / "derivatives"
assert deriv_dir.exists(), f"Could not find derivatives directory {deriv_dir}."

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(project_dir / "logs" / f"{today}_{current_script_name}.log")

# %%
# Load a dataframe with image paths to use for the template
# ---------------------------------------------------------

source_csv_dir = project_dir / "templates" / "use_for_template.csv"
df = pd.read_csv(source_csv_dir)

# take the hemi column and split it into two boolean columns
# named use_left and use_right (for easier filtering)
df["use_left"] = df["hemi"].isin(["left", "both"])
df["use_right"] = df["hemi"].isin(["right", "both"])
# Keep only the rows in which at least one of use_left or use_right is True
df = df[df[["use_left", "use_right"]].any(axis=1)]
n_subjects = len(df)

# %%
# Run the pipeline for each subject
# ---------------------------------

# Create a dictionary to store the paths to the use4template directories
# per subject. These will contain all necessary images for template building.
use4template_dirs = {}

for idx, row in tqdm(df.iterrows(), total=n_subjects):
    # Figure out input-output paths
    row = df.iloc[idx]
    subject = "sub-" + row["subject_id"]
    channel_str = "channel-" + row["color"]
    source_origin = row["orientation"]
    file_prefix = f"{subject}_{res_str}_{channel_str}"
    deriv_subj_dir = deriv_dir / subject
    deriv_subj_dir.mkdir(exist_ok=True)
    tiff_path = raw_dir / f"{file_prefix}.tif"  # original tiff image
    nii_ext = ".nii.gz"  # extension for nifti files

    logger.info(f"Starting to process {file_prefix}...")
    logger.info(f"Will save outputs to {deriv_subj_dir}/")

    # Load the manually aligned nifti image
    nii_path = file_path_with_suffix(
        deriv_subj_dir / tiff_path.name, "_orig-asr_aligned", new_ext=nii_ext
    )
    image_ants = ants.image_read(nii_path.as_posix())
    logger.debug(
        f"Read image with shape: {image_ants.shape} from {nii_path.name}."
    )

    # Load the manually adjusted brain mask for the image
    mask_path = file_path_with_suffix(
        deriv_subj_dir / tiff_path.name,
        "_orig-asr_label-brain_aligned",
        new_ext=nii_ext,
    )
    mask = ants.image_read(mask_path.as_posix())
    logger.debug(
        f"Read brain mask with shape: {mask.shape} from {mask_path.name}."
    )

    # Load the halves mask for the image (also pre-generated via the widget)
    # This will be later used for diagnostic plotting
    halves_mask_path = file_path_with_suffix(
        deriv_subj_dir / tiff_path.name,
        "_orig-asr_label-halves_aligned",
        new_ext=nii_ext,
    )
    halves_mask = ants.image_read(halves_mask_path.as_posix())
    logger.debug(
        f"Read halves mask with shape: {halves_mask.shape} from "
        f"{halves_mask_path.name}."
    )

    # Bias field correction (to homogenise intensities)
    image_n4 = ants.n4_bias_field_correction(image_ants)
    image_n4_masked_numpy = image_n4.numpy() * mask.numpy()
    image_n4_path = file_path_with_suffix(nii_path, "_N4")
    image_n4 = image_n4.new_image_like(image_n4_masked_numpy)
    ants.image_write(image_n4, image_n4_path.as_posix())
    logger.debug(
        f"Created N4 bias field corrected image as {image_n4_path.name}."
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

    # Plot the halves mask over the aligned image to check the split
    output_prefix = file_path_with_suffix(nii_path, "_N4_aligned_", new_ext="")
    ants.plot(
        image_n4,
        halves_mask,
        overlay_alpha=0.5,
        axis=1,
        title="Aligned image split into right and left halves",
        filename=output_prefix.as_posix() + "halves-overlay.png",
    )
    logger.debug("Plotted overlays to visually check alignment.")

    # Generate arrays for template construction and save as niftis
    use4template_dir = Path(output_prefix.as_posix() + "padded_use4template")
    # if it exists, delete existing files in it
    if use4template_dir.exists():
        logger.warning(f"Removing existing files in {use4template_dir}.")
        for file in use4template_dir.glob("*"):
            file.unlink()
    use4template_dir.mkdir(exist_ok=True)

    array_dict = generate_arrays_4template(
        subject, image_n4.numpy(), mask.numpy(), pad=2
    )
    save_array_dict_to_nii(array_dict, use4template_dir, vox_sizes)
    use4template_dirs[subject] = use4template_dir
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
    "asym-brain": [],
    "asym-mask": [],
    "sym-brain": [],
    "sym-mask": [],
    "hemi-brain": [],
    "hemi-mask": [],
}

for _, row in df.iterrows():
    subject = "sub-" + row["subject_id"]
    use4template_dir = use4template_dirs[subject]

    if row["hemi"] == "both":
        # Add paths for the asymmetric brain template
        for label in ["brain", "mask"]:
            filepath_lists[f"asym-{label}"].append(
                use4template_dir / f"{subject}_asym-{label}.nii.gz"
            )

    if row["use_right"]:
        for label in ["brain", "mask"]:
            # Add paths for the symmetric brain template
            filepath_lists[f"sym-{label}"].append(
                use4template_dir / f"{subject}_right-sym-{label}.nii.gz"
            )
            # Add paths for the hemispheric template
            filepath_lists[f"hemi-{label}"].append(
                use4template_dir / f"{subject}_right-hemi-{label}.nii.gz"
            )

    if row["use_left"]:
        for label in ["brain", "mask"]:
            # Add paths for the symmetric brain template
            filepath_lists[f"sym-{label}"].append(
                use4template_dir / f"{subject}_left-sym-{label}.nii.gz"
            )
            # Add paths for the hemispheric template
            filepath_lists[f"hemi-{label}"].append(
                use4template_dir / f"{subject}_left-hemi-xflip-{label}.nii.gz"
            )

# %%
# Save the file paths to text files, each in a separate directory

for key, paths in filepath_lists.items():
    kind, label = key.split("-")  # e.g. "asym" and "brain"
    n_images = len(paths)
    template_name = f"template_{kind}_{res_str}_n-{n_images}"
    template_dir = project_dir / "templates" / template_name
    template_dir.mkdir(exist_ok=True)
    np.savetxt(template_dir / f"{label}_paths.txt", paths, fmt="%s")

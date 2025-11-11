"""
Prepare low-resolution BlackCap images for template construction
================================================================
The following operations are performed on the lowest-resolution images to
prepare them for template construction:
- Import each image as tiff, re-orient to ASR and save as nifti
- Perform N4 Bias field correction using ANTs
- Generate brain mask based on N4-corrected image
- Rigid-register the re-oriented image to an already aligned target (with ANTs)
- Transform the image and mask into the aligned space
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
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image.save import save_as_asr_nii
from loguru import logger
from tqdm import tqdm

from brainglobe_template_builder.io import (
    file_path_with_suffix,
    load_tiff,
)
from brainglobe_template_builder.preproc.masking import create_mask
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
lowres = 50
# String to identify the resolution in filenames
res_str = f"res-{lowres}um"
# Define voxel sizes in mm (for Nifti saving)
vox_sizes = [lowres * 1e-3] * 3  # in mm

# Prepare directory structure
atlas_dir = Path("/ceph/neuroinformatics/neuroinformatics/atlas-forge")
species_id = "BlackCap"
species_dir = atlas_dir / species_id
deriv_dir = species_dir / "derivatives"
assert deriv_dir.exists(), f"Could not find derivatives directory {deriv_dir}."

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(species_dir / "logs" / f"{today}_{current_script_name}.log")

# %%
# Define registration target
# --------------------------
# We need three images, all in ASR orientation and nifti format:
# 1. The reference image aligned to the mid-sagittal plane
# 2. The brain mask of the reference image
# 3. The mask of the halves of the reference image (left and right)

# Here we used the previously generated template as a target
# We have manually aligned the template to the mid-sagittal plane with napari
# and prepared the masks accordingly.

target_dir = species_dir / "templates" / "template_asym_res-50um_n-3"
target_image = ants.image_read(
    (target_dir / "template_orig-asr_aligned.nii.gz").as_posix()
)
target_mask = ants.image_read(
    (target_dir / "template_orig-asr_label-brain_aligned.nii.gz").as_posix()
)
target_halves_mask = ants.image_read(
    (target_dir / "template_orig-asr_label-halves_aligned.nii.gz").as_posix()
)

# %%
# Load a dataframe with image paths to use for the template
# ---------------------------------------------------------

source_csv_dir = species_dir / "templates" / "use_for_template.csv"
df = pd.read_csv(source_csv_dir)
n_subjects = len(df)

# take the hemi column and split it into two boolean columns
# named use_left and use_right (for easier filtering)
df["use_left"] = df["hemi"].isin(["left", "both"])
df["use_right"] = df["hemi"].isin(["right", "both"])

df.head(n_subjects)

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
    file_prefix = f"{subject}_{res_str}_{channel_str}"
    deriv_subj_dir = deriv_dir / subject
    tiff_path = deriv_subj_dir / f"{file_prefix}.tif"

    logger.info(f"Starting to process {file_prefix}...")
    logger.info(f"Will save outputs to {deriv_subj_dir}/")

    # Load the image
    image = load_tiff(tiff_path)
    logger.debug(f"Loaded image {tiff_path.name} with shape: {image.shape}.")

    # Reorient the image to ASR
    source_origin = ["P", "S", "R"]
    target_origin = ["A", "S", "R"]
    source_space = AnatomicalSpace(source_origin, shape=image.shape)
    image_asr = source_space.map_stack_to(target_origin, image)
    logger.debug(f"Reoriented image from {source_origin} to {target_origin}.")
    logger.debug(f"Reoriented image shape: {image_asr.shape}.")

    # Save the reoriented image as nifti
    nii_path = file_path_with_suffix(tiff_path, "_orig-asr", new_ext=".nii.gz")
    save_as_asr_nii(image_asr, vox_sizes, nii_path)
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

    # Rigid-register the reoriented image to an already aligned target
    output_prefix = file_path_with_suffix(nii_path, "_N4_aligned_", new_ext="")

    xfm = ants.registration(
        fixed=target_image,
        moving=image_n4,
        mask=target_mask,
        moving_mask=mask,
        type_of_transform="Rigid",
        initial_transform=None,
        verbose=False,
        outprefix=output_prefix.as_posix(),
    )
    logger.debug(
        "Aligned the reoriented image to the target via rigid registration."
    )
    aligned_image = xfm["warpedmovout"]

    # Ensure we only grab the correct transform (forward rigid)
    rigid_transform = [
        file_path
        for file_path in xfm["fwdtransforms"]
        if file_path.endswith("0GenericAffine.mat")
    ]
    assert len(rigid_transform) == 1, "Expected one affine transform."

    # Transform the brain mask to the aligned image space
    aligned_mask = ants.apply_transforms(
        fixed=target_image,
        moving=mask,
        transformlist=rigid_transform,
        interpolator="nearestNeighbor",
    )
    logger.debug("Transformed brain mask to aligned space.")

    # Plot the aligned image over the target to check registration
    ants.plot(
        target_image,
        aligned_image,
        overlay_alpha=0.5,
        overlay_cmap="plasma",
        axis=1,
        title="Aligned image over target (rigid registration)",
        filename=output_prefix.as_posix() + "target-overlay.png",
    )
    # Plot the halves mask over the aligned image to check the split
    ants.plot(
        aligned_image,
        target_halves_mask,
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
        subject, aligned_image.numpy(), aligned_mask.numpy(), pad=2
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
    template_dir = species_dir / "templates" / template_name
    template_dir.mkdir(exist_ok=True)
    np.savetxt(template_dir / f"{label}_paths.txt", paths, fmt="%s")

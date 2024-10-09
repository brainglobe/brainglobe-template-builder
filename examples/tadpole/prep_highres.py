"""
Prepare high-resolution BlackCap images for template construction
================================================================
The following operations are performed on the images to
prepare them for template construction:
- Reorient the images to ASR orientation and convert to Nifti format
- Perform N4 bias field correction to homogenise intensities
- Align the images to the midline using pre-computed rigid transforms
- Upsample the brain mask from low resolution to high resolution
- Mask the brain and crop the image to the mask extents with some padding
- Generate arrays for template construction and save as Nifti files
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
from loguru import logger
from tqdm import tqdm

from brainglobe_template_builder.io import (
    file_path_with_suffix,
    load_tiff,
    save_as_asr_nii,
)
from brainglobe_template_builder.preproc.splitting import (
    generate_arrays_4template,
    save_array_dict_to_nii,
)
from brainglobe_template_builder.preproc.transform_utils import apply_transform

# %%
# Setup
# -----
# Define some global variables, including paths to input and output directories
# and set up logging.

# Define voxel size (microns) of the low resolution images (already prepped)
lowres = 12
# Define voxel size (microns) of the high resolution images (to be prepped)
highres = 6

# String to identify the resolution in filenames
lowres_str = f"res-{lowres}um"
highres_str = f"res-{highres}um"
# Define voxel sizes in mm (for Nifti saving)
lowres_vox_sizes = [lowres * 1e-3] * 3  # in mm
highres_vox_sizes = [highres * 1e-3] * 3  # in mm

# Prepare directory structure
atlas_forge_path = Path("/media/ceph-niu/neuroinformatics/atlas-forge")
atlas_dir = (
    atlas_forge_path / "Tadpole" / "tadpole-template-starter" / "old-tadpole"
)
assert atlas_dir.exists(), f"Could not find atlas directory {atlas_dir}."
raw_dir = atlas_dir / "rawdata"
deriv_dir = atlas_dir / "derivatives"
assert raw_dir.exists(), f"Could not find raw data directory {raw_dir}."
assert deriv_dir.exists(), f"Could not find derivatives directory {deriv_dir}."

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(atlas_dir / "logs" / f"{today}_{current_script_name}.log")


# %%
# Load a dataframe with image paths to use for the template
# ---------------------------------------------------------

source_csv_dir = atlas_dir / "templates" / "use_for_template.csv"
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

    subject = "sub-" + row["subject_id"]
    channel_str = "channel-" + row["color"]
    source_origin = row["orientation"]
    file_prefix = f"{subject}_{highres_str}_{channel_str}"
    file_prefix_lowres = f"{subject}_{lowres_str}_{channel_str}"
    deriv_subj_dir = deriv_dir / subject
    tiff_path = raw_dir / f"{file_prefix}.tif"

    logger.info(f"Starting to process {file_prefix}...")
    logger.info(f"Will save outputs to {deriv_subj_dir}/")

    # Load the image
    image = load_tiff(tiff_path)
    logger.debug(f"Loaded image {tiff_path.name} with shape: {image.shape}.")

    # Reorient the image to ASR
    target_origin = "ASR"
    source_space = AnatomicalSpace(source_origin, shape=image.shape)
    image_asr = source_space.map_stack_to(target_origin, image)
    logger.debug(f"Reoriented image from {source_origin} to {target_origin}.")
    logger.debug(f"Reoriented image shape: {image_asr.shape}.")

    # Save the reoriented image as nifti
    nii_path = deriv_subj_dir / f"{file_prefix}_orig-asr.nii.gz"
    save_as_asr_nii(image_asr, highres_vox_sizes, nii_path)
    logger.debug(f"Saved reoriented image as {nii_path.name}.")

    # Bias field correction (to homogenise intensities)
    image_ants = ants.image_read(nii_path.as_posix())
    image_n4 = ants.n4_bias_field_correction(image_ants)
    image_n4_path = file_path_with_suffix(nii_path, "_N4")
    # ants.image_write(image_n4, image_n4_path.as_posix())
    logger.debug("Performed N4 bias field correction.")

    # Read midline alignment transform (as written by the napari widget)
    transform_files = list(deriv_subj_dir.glob("*_xfm-midplane.txt"))
    rigid_transform_file = [transform_files[0].as_posix()]
    assert len(rigid_transform_file) == 1, "Expected one transform."
    rigid_mat = np.loadtxt(rigid_transform_file[0])
    rigid_mat[:3, 3] *= lowres / highres  # adjust for voxel size difference
    logger.debug("Read rigid transform from file.")

    # Align image by applying the rigid transform
    aligned_image_data = apply_transform(image_n4.numpy(), rigid_mat)
    aligned_image = image_n4.new_image_like(aligned_image_data)
    logger.debug("Aligned image by applying rigid transform.")

    # Upsample the aligned brain mask from lowres to highres
    mask_lowres_path = (
        deriv_subj_dir
        / f"{file_prefix_lowres}_orig-asr_label-brain_aligned.nii.gz"
    )
    mask_lowres = ants.image_read(mask_lowres_path.as_posix())
    highres_shape = aligned_image.shape
    mask = ants.resample_image(
        mask_lowres, highres_shape, use_voxels=True, interp_type=1
    )
    mask_path = file_path_with_suffix(nii_path, "_label-brain_aligned")
    ants.image_write(mask, mask_path.as_posix())
    logger.debug(
        f"Upsampled brain mask from {mask_lowres_path.name} "
        f"to {mask_path.name}."
    )

    # Plot the mask over the image to check
    mask_plot_path = (
        deriv_subj_dir / f"{file_prefix}_orig-asr_N4_aligned_mask-overlay.png"
    )
    ants.plot(
        aligned_image,
        mask,
        overlay_alpha=0.5,
        axis=1,
        title="Brain mask over aligned image",
        filename=mask_plot_path.as_posix(),
    )
    logger.debug("Plotted overlay to visually check mask.")

    # Mask the brain (remove image background)
    aligned_masked_data = aligned_image.numpy() * mask.numpy()

    # Generate arrays for template construction and save as niftis
    output_prefix = file_path_with_suffix(
        nii_path, "_N4_aligned_masked_cropped_padded", new_ext=""
    )
    use4template_dir = Path(output_prefix.as_posix() + "_use4template")
    # if it exists, delete existing files in it
    if use4template_dir.exists():
        logger.warning(f"Removing existing files in {use4template_dir}.")
        for file in use4template_dir.glob("*"):
            file.unlink()
    use4template_dir.mkdir(exist_ok=True)

    array_dict = generate_arrays_4template(
        subject, aligned_masked_data, mask.numpy(), crop=True, padding=10
    )
    save_array_dict_to_nii(array_dict, use4template_dir, highres_vox_sizes)

    logger.info(
        f"Saved images for template construction in {use4template_dir}."
    )
    logger.info(f"Finished processing {file_prefix}.")

    # Store the path to the use4template directory for this subject
    use4template_dirs[subject] = use4template_dir


# %%
# Generate lists of file paths for template construction
# -----------------------------------------------------
# Use the paths to the use4template directories to generate lists of file paths
# for the template construction pipeline. Three kinds of template will be
# generated, and each needs the corresponging brain image and mask files:
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

# Modify the atlas-forge path before writing the paths
# if we want to use these paths on a different machine (HPC cluster)
hpc_atlas_forge_path = Path(
    "/ceph/neuroinformatics/neuroinformatics/atlas-forge"
)

for key, paths in filepath_lists.items():
    kind, label = key.split("-")  # e.g. "asym" and "brain"
    n_images = len(paths)
    template_name = f"template_{kind}_{highres_str}_n-{n_images}"
    template_dir = atlas_dir / "templates" / template_name
    template_dir.mkdir(exist_ok=True)

    paths = [
        p.as_posix().replace(
            atlas_forge_path.as_posix(), hpc_atlas_forge_path.as_posix()
        )
        for p in paths
    ]
    np.savetxt(template_dir / f"{label}_paths.txt", paths, fmt="%s")

# %%

"""
Align BlackCap images to mid-sagittal plane and split hemispheres
=================================================================
The following operations are performed:
- Import image as tiff and re-orient to ASR
- Generate brain mask
- Save re-oriented image and brain mask as niftis
- Rigid-register the re-oriented image to an already aligned target
- Split the image into hemispheres and reflect the left hemisphere
"""

# %%
# Imports
# -------
from datetime import date
from pathlib import Path

import ants
import numpy as np
import pandas as pd
from brainglobe_space import AnatomicalSpace
from loguru import logger

from brainglobe_template_builder.io import load_tiff, save_nii
from brainglobe_template_builder.preproc.masking import create_mask

# %%
# Set up output directory and logging
# ------------------------------------

# Create the output directory
atlas_dir = Path("/media/ceph-niu/neuroinformatics/atlas-forge")
species_id = "BlackCap"
output_dir = atlas_dir / species_id
deriv_dir = output_dir / "derivatives"
assert deriv_dir.exists(), f"Could not find derivatives directory {deriv_dir}."

# Set up logging
today = date.today()
current_script_name = "BlackCap_align_and_split"
logger.add(output_dir / f"{today}_{current_script_name}.log")

# %%
# Define registration target
# --------------------------
# We need three images, all in ASR orientation and nifti format:
# 1. The reference image aligned to the mid-sagittal plane
# 2. The brain mask of the reference image
# 3. The mask of the halves of the reference image (left and right)


target_dir = deriv_dir / "template_asym_n=3"
target_image_path = target_dir / "template_orig-asr_aligned.nii.gz"
target_image = ants.image_read(target_image_path.as_posix())
target_mask_path = target_dir / "template_orig-asr_label-brain_aligned.nii.gz"
target_mask = ants.image_read(target_mask_path.as_posix())
target_halves_mask_path = (
    target_dir / "template_orig-asr_label-halves_aligned.nii.gz"
)
target_halves_mask = ants.image_read(target_halves_mask_path.as_posix())


# %%
# Load a dataframe with images to use for the template
# ----------------------------------------------------

source_csv_dir = deriv_dir / "use_for_template.csv"
df = pd.read_csv(source_csv_dir)
n_subjects = len(df)
df.head(n_subjects)

# %%
# Define helper functions
# -----------------------


def file_path_with_suffix(path: Path, suffix: str, new_ext=None) -> Path:
    """
    Return a new path with the given suffix added before the extension.

    Parameters
    ----------
    path : Path
        The file path to modify.
    suffix : str
        The suffix to add before the extension.
    new_ext : str, optional
        If given, replace the current extension with this one.
        Should include the leading period.

    Returns
    -------
    str
        The new path to the file with the given suffix.

    """
    suffixes = "".join(path.suffixes)
    pure_stem = str(path.stem).rstrip(suffixes)
    if new_ext is not None:
        new_name = f"{pure_stem}{suffix}{new_ext}"
    else:
        new_name = f"{pure_stem}{suffix}{suffixes}"
    return path.with_name(new_name)


# %%
# Run the pipeline for each subject
# ---------------------------------

# Define the resolution and voxel sizes (same for all images)
MICRONS = 50
res_str = f"res-{MICRONS}um"
vox_sizes = [MICRONS * 1e-3] * 3  # in mm

# %%

# for idx, row in tqdm(df.iterrows(), total=n_subjects):
row = df.iloc[0]
subject_str = "sub-" + row["subject_id"]
channel_str = "channel-" + row["color"]
file_prefix = f"{subject_str}_{res_str}_{channel_str}"
deriv_subj_dir = deriv_dir / subject_str
tiff_path = deriv_subj_dir / f"{file_prefix}.tif"

logger.info(f"Starting to process {file_prefix}...")
logger.info(f"Will save outputs to {deriv_subj_dir}/")

# %%

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
save_nii(image_asr, vox_sizes, nii_path, kind="image")
logger.debug(f"Saved reoriented image as {nii_path.name}.")

# %%

# Bias field correction (to homogenise intensities)
image_ants = ants.image_read(nii_path.as_posix())
image_n4 = ants.n4_bias_field_correction(image_ants)

# Generate a brain mask based on the N4-corrected image
mask_data = create_mask(
    image_n4.numpy(),
    gauss_sigma=3,
    threshold_method="triangle",
    closing_size=5,
)
mask = image_n4.new_image_like(mask_data.astype(np.uint8))
logger.debug(f"Generated brain mask with shape: {mask.shape}.")

# Plot the mask over the image to check
mask_plot_path = deriv_subj_dir / f"{file_prefix}_orig-asr_n4_mask_overlay.png"
ants.plot(
    image_n4,
    mask,
    overlay_alpha=0.5,
    axis=1,
    title="Brain mask over image",
    filename=mask_plot_path.as_posix(),
)
logger.debug("Plotted overlay to visually check mask.")

# %%

# Save the N4-corrected image and brain mask as niftis
ants.image_write(image_n4, file_path_with_suffix(nii_path, "_N4").as_posix())
ants.image_write(
    mask, file_path_with_suffix(nii_path, "_N4_label_brain").as_posix()
)

# %%

# Rigid-register the reoriented image to an already aligned target
output_prefix = file_path_with_suffix(
    nii_path, "_N4_aligned", new_ext=""
).as_posix()
xfm = ants.registration(
    fixed=target_image,
    moving=image_n4,
    mask=target_mask,  # in target space
    moving_mask=mask,
    type_of_transform="Rigid",
    initial_transform=None,
    verbose=False,
    outprefix=output_prefix,
)
logger.debug(
    "Aligned the reoriented image to the target via rigid registration."
)
aligned_image = xfm["warpedmovout"]
aligned_image_path = Path(output_prefix + ".nii.gz")
ants.image_write(aligned_image, aligned_image_path.as_posix())
logger.debug(f"Saved aligned image as {aligned_image_path.name}.")

# Transform the brain mask to the aligned image space
aligned_mask = ants.apply_transforms(
    fixed=aligned_image,
    moving=mask,
    transformlist=xfm["fwdtransforms"],
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
    filename=output_prefix + "_target_overlay.png",
)
# Plot the halves mask over the aligned image to check the split
ants.plot(
    aligned_image,
    target_halves_mask,
    overlay_alpha=0.5,
    axis=1,
    title="Aligned image split into right and left halves",
    filename=output_prefix + "_halves_overlay.png",
)
logger.debug("Plotted overlays to visually check alignment.")

# Split the aligned image and its mask into hemispheres
aligned_brain_data = aligned_image.numpy()
aligned_mask_data = aligned_mask.numpy().astype(np.uint8)
z, y, x = aligned_brain_data.shape
# Create the slices for the first and second halves along the x axis (last)
right_half = (slice(None), slice(None), slice(0, x // 2))
left_half = (slice(None), slice(None), slice(x // 2, x))

# right hemisphere and its reflection
right_hemi_brain = aligned_brain_data[right_half]
right_hemi_mask = aligned_mask_data[right_half]
right_hemi_brain_xflip = np.flip(right_hemi_brain, axis=-1)
right_hemi_mask_xflip = np.flip(right_hemi_mask, axis=-1)

# left hemisphere and its reflection
left_hemi_brain = aligned_brain_data[left_half]
left_hemi_mask = aligned_mask_data[left_half]
left_hemi_brain_xflip = np.flip(left_hemi_brain, axis=-1)
left_hemi_mask_xflip = np.flip(left_hemi_mask, axis=-1)

# right-symmetric and left-symmetric brains
right_sym_brain = np.dstack([right_hemi_brain, right_hemi_brain_xflip])
right_sym_mask = np.dstack([right_hemi_mask, right_hemi_mask_xflip])
left_sym_brain = np.dstack([left_hemi_brain_xflip, left_hemi_brain])
left_sym_mask = np.dstack([left_hemi_mask_xflip, left_hemi_mask])

save_dict = {
    "asym-brain": aligned_brain_data,
    "asym-mask": aligned_mask_data,
    "right-hemi-brain": right_hemi_brain,
    "right-hemi-mask": right_hemi_mask,
    "right-hemi-brain-xflip": right_hemi_brain_xflip,
    "right-hemi-mask-xflip": right_hemi_mask_xflip,
    "left-hemi-brain": left_hemi_brain,
    "left-hemi-mask": left_hemi_mask,
    "left-hemi-brain-xflip": left_hemi_brain_xflip,
    "left-hemi-mask-xflip": left_hemi_mask_xflip,
    "right-sym-brain": right_sym_brain,
    "right-sym-mask": right_sym_mask,
    "left-sym-brain": left_sym_brain,
    "left-sym-mask": left_sym_mask,
}

# create directory for files intended for template construction
template_dir = Path(output_prefix + "_4template")
template_dir.mkdir(exist_ok=True)
logger.info(
    f"Files for template construction will be saved to {template_dir}."
)
for key, data in save_dict.items():
    save_path = template_dir / f"{key}.nii.gz"
    if "mask" not in key:
        save_nii(data, vox_sizes, save_path)
        logger.debug(f"Saved {save_path.name}.")


# %%

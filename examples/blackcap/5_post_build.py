"""Post-processing after building the BlackCap template.

This script is run after the BlackCap template has been built. It performs
the following steps:
1. Select the appropriate template and pad to generate the atlas reference.
2. Resample the annotated subject image and mask to the atlas resolution.
3. Generate brain masks for the atlas reference and the annotated subject.
4. Register the annotated subject image to the atlas reference image.
5. Transform the annotations to the atlas reference image space.
"""

# %%
# Imports
# -------
import os
import subprocess
from datetime import date
from pathlib import Path

import ants
import numpy as np
from brainglobe_utils.IO.image.save import save_as_asr_nii
from loguru import logger

from brainglobe_template_builder.preproc.masking import create_mask

# %%
# Setup
# -----

# Species-specific atlas-forge directory
atlas_dir = Path("/ceph/neuroinformatics/neuroinformatics/atlas-forge")
species_id = "BlackCap"
species_dir = atlas_dir / species_id
res_um = 25  # resolution in microns

# Define the path to the final template file to use as the atlas reference
res_str = f"res-{res_um}um"
template_name = f"template_sym_{res_str}_n-18"
# Voxel size in mm (as in nifti)
res_mm = res_um * 1e-3
vox_sizes = [res_mm, res_mm, res_mm]

template_build_dir = species_dir / "templates" / template_name
template_final_dir = template_build_dir / "final" / "average"
template_file = template_final_dir / "template_sharpen_shapeupdate.nii.gz"
assert (
    template_final_dir.is_dir()
), f"Directory {template_final_dir} does not exist."
assert template_file.is_file(), f"File {template_file} does not exist."

# Define paths to the annotation file and the corresponding subject's image
annot_subj = "sub-BC74"
annot_dir = species_dir / "rawdata" / annot_subj / "annotations"
annot_img_file = (
    annot_dir
    / "1001010_ds_SW_BC74white_220217_120749_10_10_ch03_chan_3_green_raw_oriented.nii"  # noqa
)
assert annot_img_file.is_file(), f"File {annot_img_file} does not exist."
annot_labels_file = (
    annot_dir / "BC74white_100um_annoations_IM_200522_16_Feb_2024.nii"
)
assert annot_labels_file.is_file(), f"File {annot_labels_file} does not exist."

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(species_dir / "logs" / f"{today}_{current_script_name}.log")

# Create the output directory
output_dir = species_dir / "templates" / template_name / "for_atlas"
output_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Post-processing {species_id} {template_name}.")
logger.info(f"Outputs will be saved to {output_dir}.")


# %%
# Create atlas reference image
# ----------------------------
# The atlas reference image is the final template with some zero-padding
# (to avoid masks/labels being cut off at the edges).

n_pad = 6  # Number of planes to pad on each side

# Load the template file and add n_pad planes of zero-padding to each side
template = ants.image_read(template_file.as_posix())
template_padded = np.pad(template.numpy(), pad_width=n_pad, mode="constant")
logger.debug(f"Template zero-padded to shape: {template_padded.shape}.")

# Save the padded template to the output directory
# Henceforth we will refer to it as the atlas reference image.
atlas_reference_file = output_dir / f"reference_{res_str}_image.nii.gz"
save_as_asr_nii(template_padded, vox_sizes, atlas_reference_file)
logger.debug(f"Saved atlas reference image as {atlas_reference_file.name}.")

# Plot the atlas reference image to check
atlas_reference = ants.image_read(atlas_reference_file.as_posix())
ants.plot(
    atlas_reference,
    axis=1,
    title="Atlas reference image",
    filename=output_dir / f"reference_{res_str}_image.png",
)

# %%
# Resample annotated subject image and mask to the atlas reference resolution
# ---------------------------------------------------------------------------

annot_img = ants.image_read(annot_img_file.as_posix())
annot_labels = ants.image_read(annot_labels_file.as_posix())

annot_img_res = ants.resample_image(
    image=annot_img,
    resample_params=vox_sizes,  # Target voxel sizes
    use_voxels=False,
    interp_type=4,  # B-spline interpolation
)
annot_labels_res = ants.resample_image(
    image=annot_labels,
    resample_params=vox_sizes,  # Target voxel sizes
    use_voxels=False,
    interp_type=1,  # Nearest neighbor interpolation
)

annot_img_res_file = output_dir / f"{annot_subj}_{res_str}_image.nii.gz"
ants.image_write(annot_img_res, annot_img_res_file.as_posix())
logger.debug(f"Resampled annotated {annot_subj} image to {res_str} ")
logger.debug(f"and saved as {annot_img_res_file.name}.")

annot_labels_res_file = output_dir / f"{annot_subj}_{res_str}_labels.nii.gz"
ants.image_write(annot_labels_res, annot_labels_res_file.as_posix())
logger.debug(f"Resampled annotated {annot_subj} labels to {res_str} ")
logger.debug(f"and saved as {annot_labels_res_file.name}.")


# %%
# Generate brain masks
# --------------------
# Two masks in the atlas reference image space:
# - A loose mask for registration, e.g. "reference_res-50um_mask-4reg"
# - A tight mask tos serve as root label, e.g. "reference_res-50um_mask-brain"
# One mask in the resampled annotated subject image space:
# - A loose mask for registration, e.g. "sub-BC74_res-50um_mask-4reg"

# Loose mask in atlas reference space (for registration)
reg_mask_arr = create_mask(
    atlas_reference.numpy(),
    gauss_sigma=3,
    threshold_method="triangle",
    closing_size=5,
    erode_size=0,
)
reg_mask = atlas_reference.new_image_like(reg_mask_arr.astype(np.uint8))
reg_mask_file = output_dir / f"reference_{res_str}_mask-4reg.nii.gz"
ants.image_write(reg_mask, reg_mask_file.as_posix())

# Tight mask in atlas reference space (for root label)
brain_mask_arr = create_mask(
    atlas_reference.numpy(),
    gauss_sigma=2,
    threshold_method="triangle",
    closing_size=4,
    erode_size=4,
)
brain_mask = atlas_reference.new_image_like(brain_mask_arr.astype(np.uint8))
brain_mask_file = output_dir / f"reference_{res_str}_mask-brain.nii.gz"
ants.image_write(brain_mask, brain_mask_file.as_posix())

# Loose mask in annotated subject image space (for registration)
subj_reg_mask_arr = create_mask(
    annot_img_res.numpy(),
    gauss_sigma=3,
    threshold_method="triangle",
    closing_size=5,
    erode_size=0,
)
subj_reg_mask = annot_img_res.new_image_like(
    subj_reg_mask_arr.astype(np.uint8)
)
subj_reg_mask_file = output_dir / f"{annot_subj}_{res_str}_mask-4reg.nii.gz"
ants.image_write(subj_reg_mask, subj_reg_mask_file.as_posix())

logger.debug("Generated brain masks.")

# Plot the masks over the reference image to check
for bg_name, bg_img, mask_name, mask_img in zip(
    ("reference", "reference", annot_subj),
    (atlas_reference, atlas_reference, annot_img_res),
    ("4reg", "brain", "4reg"),
    (reg_mask, brain_mask, subj_reg_mask),
):
    mask_plot_path = (
        output_dir / f"{bg_name}_{res_str}_mask-{mask_name}_overlay.png"
    )
    ants.plot(
        bg_img,
        mask_img,
        overlay_alpha=0.5,
        axis=1,
        title=f"mask-{mask_name} over {bg_name} image",
        filename=mask_plot_path.as_posix(),
    )
logger.debug("Plotted overlays to visually check masks.")


# %%
# Register annotated subject image to the atlas reference image
# --------------------------------------------------------------

output_prefix = output_dir / f"{annot_subj}_to_reference_{res_str}_"

# Call the antsRegistration_affine_SyN.sh script with the provided parameters

cmd = "antsRegistration_affine_SyN.sh "
cmd += f"--moving-mask {subj_reg_mask_file.as_posix()} "
cmd += f"--fixed-mask {reg_mask_file.as_posix()} "
cmd += "--clobber "  # Overwrite existing files
cmd += f"{annot_img_res_file.as_posix()} "  # moving image
cmd += f"{atlas_reference_file.as_posix()} "  # fixed image
cmd += f"{output_prefix.as_posix()}"  # output prefix

logger.debug(f"Running the following ANTs registration script: {cmd}.")
subprocess.run(cmd, shell=True, check=True)
logger.debug(
    f"Registered {annot_subj} {res_str} image to atlas reference image."
)

# %%
# Transform the annotations to the atlas reference image space
# -------------------------------------------------------------

# Order of transforms is as in linear algebra (last applied first)
transforms = [
    output_prefix.as_posix() + "1Warp.nii.gz",
    output_prefix.as_posix() + "0GenericAffine.mat",
]

# Apply the transforms to the resampled annotated subject image
aligned_img = ants.apply_transforms(
    fixed=atlas_reference,
    moving=annot_img_res,
    transformlist=transforms,
    interpolator="bSpline",
)
# Save the transformed image to the output directory
aligned_img_file = (
    output_dir / f"{annot_subj}_{res_str}_image_aligned-to-reference.nii.gz"
)
ants.image_write(aligned_img, aligned_img_file.as_posix())
logger.debug(
    f"Transformed {annot_subj} {res_str} image to atlas reference image space."
)
logger.debug(f"Saved transformed image as {aligned_img_file.name}.")


aligned_labels = ants.apply_transforms(
    fixed=atlas_reference,
    moving=annot_labels_res,
    transformlist=transforms,
    interpolator="genericLabel",
)
# Save the transformed annotations to the output directory
aligned_labels_file = (
    output_dir / f"{annot_subj}_{res_str}_labels_aligned-to-reference.nii.gz"
)
ants.image_write(aligned_labels, aligned_labels_file.as_posix())
logger.debug(
    f"Transformed {annot_subj} {res_str} annotations to atlas reference image."
)
logger.debug(f"Saved transformed annotations as {aligned_labels_file.name}.")


# Plot the aligned image over the reference image to check registration
ants.plot(
    atlas_reference,
    aligned_img,
    overlay_alpha=0.5,
    overlay_cmap="plasma",
    vminol=-250,
    vmaxol=7000,
    axis=1,
    title=f"{annot_subj} {res_str} image aligned to the atlas reference",
    filename=output_prefix.as_posix() + "image-overlay.png",
)

# Plot the aligned annotation over the reference image to check registration
ants.plot(
    atlas_reference,
    aligned_labels,
    overlay_alpha=0.5,
    overlay_cmap="tab20",
    axis=1,
    title=f"{annot_subj} {res_str} annotations aligned to the atlas reference",
    filename=output_prefix.as_posix() + "labels-overlay.png",
)
logger.debug("Plotted overlays to visually check registration.")

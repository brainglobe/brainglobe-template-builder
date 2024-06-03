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
# Run the pipeline for each subject
# ---------------------------------

MICRONS = 50
res_str = f"res-{MICRONS}um"
vox_sizes = [MICRONS * 1e-3] * 3  # in mm


# for idx, row in tqdm(df.iterrows(), total=n_subjects):
for idx, row in df.iloc[:1].iterrows():
    subject_str = "sub-" + row["subject_id"]
    channel_str = "channel-" + row["color"]
    filename = f"{subject_str}_{res_str}_{channel_str}.tif"
    deriv_subj_dir = deriv_dir / subject_str
    tiff_path = deriv_subj_dir / filename

    logger.info(f"Processing {filename}...")
    logger.info(f"Will save outputs to {deriv_subj_dir}.")

    # Load the image
    image = load_tiff(tiff_path)
    logger.info(f"Loaded image from {tiff_path}.")
    logger.info(f"Image shape: {image.shape}.")

    # Reorient the image to ASR
    source_origin = ["Posterior", "Superior", "Right"]
    target_origin = ["Anterior", "Superior", "Right"]
    source_space = AnatomicalSpace(source_origin, shape=image.shape)
    image_asr = source_space.map_stack_to(target_origin, image)
    logger.info(f"Reoriented image to {target_origin}.")

    # Save the reoriented image as nifti
    nii_path = tiff_path.with_name(tiff_path.stem + "_orig-asr.nii.gz")
    save_nii(image_asr, vox_sizes, nii_path)
    logger.info(f"Saved reoriented image as {nii_path.name}.")

    # Generate a brain mask
    mask = create_mask(
        image_asr, gauss_sigma=3, threshold_method="triangle", closing_size=5
    )
    logger.info("Generated brain mask.")

    # Save the brain mask as nifti
    mask_path = nii_path.with_name(
        nii_path.stem.split(".")[0] + "_label-brain.nii.gz"
    )
    save_nii(mask, vox_sizes, mask_path)
    logger.info(f"Saved brain mask as {mask_path.name}.")

    # Plot the mask over the image to check
    image_asr_obj = ants.image_read(nii_path.as_posix())
    mask_obj = ants.image_read(mask_path.as_posix())
    ants.plot(
        image_asr_obj,
        mask_obj,
        overlay_alpha=0.5,
        overlay_cmap="plasma",
        axis=1,
        title="Brain mask over image",
        filename=nii_path.with_name(
            mask_path.stem.split(".")[0] + "_overlay.png"
        ).as_posix(),
    )

    # Run N4BiasFieldCorrection on the reoriented image
    image_asr_n4 = ants.n4_bias_field_correction(image_asr_obj, mask=mask_obj)
    image_asr_n4_path = nii_path.with_name(
        nii_path.stem.split(".")[0] + "_N4.nii.gz"
    )
    ants.image_write(image_asr_n4, image_asr_n4_path.as_posix())
    logger.info(
        "Run N4BiasFieldCorrection on the reoriented image "
        f"and saved as {image_asr_n4_path.name}."
    )

    # Rigid-register the reoriented image to an already aligned target
    output_prefix = image_asr_n4_path.with_name(
        image_asr_n4_path.stem.split(".")[0] + "_aligned"
    ).as_posix()
    results = ants.registration(
        fixed=target_image,
        moving=image_asr_n4,
        mask=target_mask,  # in target space
        moving_mask=mask_obj,
        type_of_transform="Rigid",
        initial_transform=None,
        verbose=False,
        outprefix=output_prefix,
    )
    logger.info(
        "Aligned the reoriented image to the target via rigid registration."
    )
    aligned_image = results["warpedmovout"]
    aligned_image_path = Path(output_prefix + ".nii.gz")
    ants.image_write(aligned_image, aligned_image_path.as_posix())
    logger.info(f"Saved aligned image as {aligned_image_path.name}.")

    # Transform the brain mask to the aligned image space
    aligned_mask = ants.apply_transforms(
        fixed=aligned_image,
        moving=mask_obj,
        transformlist=results["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    # Multiply the transformed mask by the halves mask halves mask
    # and then binarise to get masks for the right and left hemispheres
    aligned_hemis_mask = ants.image_clone(aligned_mask)
    aligned_hemis_mask.view()[:] = (
        aligned_mask.numpy() * target_halves_mask.numpy()
    )
    aligned_mask_path = nii_path.with_name(
        nii_path.stem.split(".")[0] + "_label-hemis_aligned.nii.gz"
    )
    ants.image_write(aligned_hemis_mask, aligned_mask_path.as_posix())
    logger.info(f"Saved aligned hemispheres mask as {aligned_mask_path.name}.")

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
    # Plot the hemi masks over the aligned image to check the split
    ants.plot(
        aligned_image,
        aligned_hemis_mask,
        overlay_alpha=0.5,
        axis=1,
        title="Aligned image split into right and left hemispheres",
        filename=output_prefix + "_hemis_overlay.png",
    )
    logger.info("Plotted overlays to visually check alignment.")


# %%

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

target_dir = deriv_dir / "template_asym_n=3"
target_image_path = target_dir / "template_orig-asr_aligned.nii.gz"
target_image = ants.image_read(target_image_path.as_posix())
target_mask_path = target_dir / "template_orig-asr_label-brain_aligned.nii.gz"
target_mask = ants.image_read(target_mask_path.as_posix())

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

    # Run N4BiasFieldCorrection on the reoriented image
    image_asr_obj = ants.image_read(nii_path.as_posix())
    mask_obj = ants.image_read(mask_path.as_posix())
    image_asr_n4 = ants.n4_bias_field_correction(image_asr_obj, mask=mask_obj)
    image_asr_n4_path = nii_path.with_name(
        nii_path.stem.split(".")[0] + "_N4.nii.gz"
    )
    ants.image_write(image_asr_n4, image_asr_n4_path.as_posix())
    logger.info(
        "Run N4BiasFieldCorrection on the reoriented image "
        f" and saved as {image_asr_n4_path.name}."
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
    logger.info("Rigid-registered the reoriented image to the target.")
    aligned_image = results["warpedmovout"]
    ants.image_write(aligned_image, output_prefix + ".nii.gz")
    logger.info(f"Saved aligned image as {output_prefix}.nii.gz.")


# %%

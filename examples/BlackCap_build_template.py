"""
Build a BlackCap template using unbiased-averaging
==================================================
This script builds a BlackCap template using the unbiased-averaging method.
It uses the BlackCap images that are created by the script
BlackCap_prep_lowres.py.
"""

# %%
# Pre-requisites
# --------------
# The BlackCap images should be pre-processed using BlackCap_prep_lowres.py
# ANTS and associated scripts should be installed as outlined in the
# README file of the [antsMultivariateTemplateConstruction repository](https://github.com/CoBrALab/optimized_antsMultivariateTemplateConstruction)  # noqa

# %%
# Imports
# -------
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

# %%
# Setup
# ---------

# Create the output directory
atlas_dir = Path("/media/ceph-niu/neuroinformatics/atlas-forge")
species_id = "BlackCap"
species_dir = atlas_dir / species_id
deriv_dir = species_dir / "derivatives"
assert deriv_dir.exists(), f"Could not find derivatives directory {deriv_dir}."

# Set up logging
today = date.today()
current_script_name = "BlackCap_build_template.py"
logger.add(deriv_dir / f"{today}_{current_script_name}.log")


# %%
# Load the list of images to use for the template

source_csv_dir = deriv_dir / "use_for_template.csv"
df = pd.read_csv(source_csv_dir)
n_subjects = len(df)
df.head(n_subjects)

# %%

sym_brains = []
sym_masks = []

for i, row in df.iterrows():
    subject_str = "sub-" + row["subject_id"]
    channel_str = "channel-" + row["color"]
    use_hemi = row["hemi"]
    deriv_subj_dir = deriv_dir / subject_str

    prefix = f"{subject_str}_res-50um_{channel_str}"
    suffix = "orig-asr_N4_aligned_use4template"
    use4template_dir = (
        deriv_subj_dir / f"{prefix}_orig-asr_N4_aligned_use4template"
    )
    assert use4template_dir.exists(), f"Could not find {use4template_dir}."

    if use_hemi == "both":
        sym_brains.append(use4template_dir / "right-sym-brain.nii.gz")
        sym_masks.append(use4template_dir / "right-sym-mask.nii.gz")
        sym_brains.append(use4template_dir / "left-sym-brain.nii.gz")
        sym_masks.append(use4template_dir / "left-sym-mask.nii.gz")
    elif use_hemi == "left":
        sym_brains.append(use4template_dir / "left-sym-brain.nii.gz")
        sym_masks.append(use4template_dir / "left-sym-mask.nii.gz")
    elif use_hemi == "right":
        sym_brains.append(use4template_dir / "right-sym-brain.nii.gz")
        sym_masks.append(use4template_dir / "right-sym-mask.nii.gz")

# %%

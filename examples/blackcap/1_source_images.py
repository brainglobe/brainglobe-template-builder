"""
Identify source images for BlackCap template
============================================
Set up project directory with all relevant data and identify source images
to be used for building the BlackCap template.
"""

# %%
# Imports
# -------
import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from brainglobe_template_builder.io import get_unique_folder_in_dir

# %%
# Set up directories and logging
# ------------------------------

# Prepare directory structure
atlas_dir = Path("/ceph/neuroinformatics/neuroinformatics/atlas-forge")
species_id = "BlackCap"
species_common_name = "Black cap"  # as in the source csv table
species_dir = atlas_dir / species_id
species_dir.mkdir(parents=True, exist_ok=True)
# Make "rawdata", "derivatives", "templates", and "logs" directories
for folder in ["rawdata", "derivatives", "templates", "logs"]:
    (species_dir / folder).mkdir(exist_ok=True)

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(species_dir / "logs" / f"{today}_{current_script_name}.log")
logger.info(f"Will save outputs to {species_dir}.")

# %%
# Load a dataframe with all SWC brains used for atlases
# ------------------------------------------------------

source_csv_dir = Path.cwd() / "data"  # relative to repo root
source_csv_path = source_csv_dir / "SWC_brain-list_for-atlases_2024-04-15.csv"
df = pd.read_csv(source_csv_path)

# Strip trailing space from str columns and from column names
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.columns = df.columns.str.strip()
# Select only data for the species of interest
df = df[df["Common name"] == species_common_name]
logger.info(f"Found {len(df)} {species_id} subjects.")

# %%
# Extract subject IDs and paths
# ----------------------------

df["subject_id"] = (
    df["Specimen ID"]
    .apply(lambda x: x[3:8].strip())
    .apply(lambda x: x[:-1] if x.endswith("w") else x)
)
# Assert that subject IDs are unique
assert len(df["subject_id"].unique()) == len(df), "Non-unique subject IDs"

data_path_col = "Data path (raw)"
df[data_path_col] = df[data_path_col].apply(
    lambda x: x.replace("nfs/ceph", "ceph")
)
df["subject_path"] = df.apply(
    lambda x: get_unique_folder_in_dir(
        Path("/" + x[data_path_col]), x["subject_id"]
    ),
    axis=1,
)

sub_ids = df["subject_id"].values
logger.info(f"Identified {sub_ids.size} unique subject IDs. {sub_ids}")
sub_paths_msg = "Subject paths:\n"
for idx, row in df.iterrows():
    sub_id = row["subject_id"]
    sub_path = row["subject_path"].as_posix()
    sub_paths_msg += f"  {sub_id}: {sub_path}\n"
logger.debug(sub_paths_msg)

# %%
# Identify all downsampled images
# -------------------------------
# We will identify all downsampled images for each subject, by iterating over
# the downsampled_stacks folders and the various channels.
# We will aggregate the image information and paths in a dataframe.

valid_colors = ["far_red", "red", "green", "blue"]  # order matters

images_list_of_dicts = []
for _, row in df.iterrows():
    sub = row["subject_id"]
    sub_path_down = row["subject_path"] / "downsampled_stacks"
    stacks = [f for f in os.listdir(sub_path_down) if f.endswith("micron")]

    for stack in sorted(stacks):
        microns = int(stack.split("_")[0])
        stack_path = sub_path_down / stack

        images = [
            f
            for f in os.listdir(stack_path)
            if f.endswith(".tif") and "_ch0" in f
        ]
        for img in sorted(images):
            ch_num = int(img.split("_ch0")[1].split("_")[0])
            ch_color = [
                c for c in valid_colors if img.split(".tif")[0].endswith(c)
            ][0]
            ch_color = ch_color if ch_color != "far_red" else "farred"
            image_id = f"sub-{sub}_res-{microns}um_channel-{ch_color}"
            images_list_of_dicts.append(
                {
                    "subject_id": sub,
                    "microns": microns,
                    "channel": ch_num,
                    "color": ch_color,
                    "image_id": image_id,
                    "image_path": stack_path / img,
                }
            )

images_df = pd.DataFrame(images_list_of_dicts)
n_img = len(images_df)
logger.info(f"Found {n_img} images across subjects, resolutions and channels.")

# %%
# Check for missing downsampled stacks
# -------------------------------------
# Logs a warning if any of the expected resolutions are missing

expected_microns = [10, 25, 50]
for sub in df["subject_id"]:
    for microns in expected_microns:
        if not (
            images_df[
                (images_df["subject_id"] == sub)
                & (images_df["microns"] == microns)
            ].shape[0]
        ):
            logger.warning(f"Subject {sub} lacks {microns} micron stack")

# %%
# Save dataframes and images to rawdata
# -------------------------------------
# Save the dataframes to the species rawdata directory.

rawdata_dir = species_dir / "rawdata"
subjects_csv = rawdata_dir / f"{today}_subjects.csv"
df.to_csv(subjects_csv, index=False)
logger.info(f"Saved subject information to csv: {subjects_csv}")

images_csv = rawdata_dir / f"{today}_images.csv"
images_df.to_csv(images_csv, index=False)
logger.info(f"Saved image information to csv: {images_csv}")

# %%
# Save images to the species rawdata directory
# (doesn't overwrite existing images).
# High-resolution images (10um) are just symlinked to avoid duplication
# of large files.

n_copied, n_symlinked = 0, 0
for idx in tqdm(images_df.index):
    row = images_df.loc[idx, :]
    sub = row["subject_id"]
    sub_dir = rawdata_dir / f"sub-{sub}"
    sub_dir.mkdir(exist_ok=True)
    microns = row["microns"]
    image_id = row["image_id"]
    img_source_path = row["image_path"]
    img_dest_path = sub_dir / f"{image_id}.tif"

    # if the destination path exists, skip
    if img_dest_path.exists():
        logger.debug(f"Skipping {img_dest_path} as it already exists.")
        continue

    # if image is 10 microns, symlink it (they are large)
    if microns == 10:
        img_dest_path.symlink_to(img_source_path)
        logger.debug(f"Symlinked {img_dest_path} to {img_source_path}")
        n_symlinked += 1
    # else copy the image
    else:
        shutil.copyfile(img_source_path, img_dest_path)
        logger.debug(f"Copied {img_source_path} to {img_dest_path}")
        n_copied += 1

logger.info(
    f"Copied {n_copied} and symlinked {n_symlinked} "
    f"images to {rawdata_dir}."
)

# %%
# Select data to use for template creation
# ----------------------------------------
# Here we select the subset of subjects to use for template creation.
# The selection is based on visual inspection of the lowest-resolution
# images (here 50um).
#
# The selected subjects are supplied as a .csv file named
# "use_for_template.csv" which should be placed in the derivatives folder.

# The resolution to use for visual inspection + preprocessing
lowres = "50um"

template_dir = species_dir / "templates"
deriv_dir = species_dir / "derivatives"

# Check for the .csv file
use_for_template_csv = template_dir / "use_for_template.csv"
if use_for_template_csv.exists():
    use_for_template = pd.read_csv(use_for_template_csv)

    assert (
        use_for_template.subject_id.is_unique
    ), "Subject IDs must be unique in use_for_template.csv"
    n_subjects_to_use = len(use_for_template)
    logger.info(
        f"Will use {n_subjects_to_use} subjects for template creation."
    )

    for _, row in use_for_template.iterrows():
        sub = row["subject_id"]
        sub_str = f"sub-{sub}"
        channel = row["color"]
        hemi = row["hemi"]
        filename = f"{sub_str}_res-{lowres}_channel-{channel}.tif"
        # Create derivatives folder for each selected subject
        (deriv_dir / sub_str).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created derivatives folder for {sub_str}")
        # copy the low-res image from rawdata to derivatives for convenience
        raw_file = rawdata_dir / sub_str / filename
        derivative_file = deriv_dir / sub_str / filename
        if derivative_file.is_file():
            logger.info(f"Skipping {derivative_file} as it already exists.")
        else:
            shutil.copyfile(raw_file, derivative_file)
            logger.info(
                f"Copied {filename} to the {sub_str} derivatives folder."
            )

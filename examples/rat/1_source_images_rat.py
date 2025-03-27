"""
Identify source images for the SWC rat template
===============================================
Set up project directory with all relevant data and identify source images
to be used for building the SWC rat brain template.

This scipt must be run on the SWC HPC cluster.
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

from brainglobe_template_builder.io import (  # type: ignore
    get_path_from_env_variable,
    get_unique_folder_in_dir,
    load_tiff,
)
from brainglobe_template_builder.plots import plot_grid, plot_orthographic

# Set up directories and logging
# ------------------------------

# Prepare directory structure
atlas_dir = get_path_from_env_variable(
    "ATLAS_FORGE_DIR", "/ceph/neuroinformatics/neuroinformatics/atlas-forge"
)

species_id = "Rat"  # the subfolder names within atlas_dir
species_dir = atlas_dir / species_id

species_dir.mkdir(parents=True, exist_ok=True)
# Make "rawdata", "derivatives", "templates", and "logs" directories
for folder in ["rawdata", "derivatives", "templates", "logs"]:
    (species_dir / folder).mkdir(exist_ok=True)

# Directory where source images are stored for this species
# This must contain subfolders for each subject
source_dir = get_path_from_env_variable(
    "ATLAS_SOURCE_DIR", "/ceph/akrami/capsid_testing/imaging/2p"
)

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(species_dir / "logs" / f"{today}_{current_script_name}.log")
logger.info(f"Will save outputs to {species_dir}.")

# %%
# Load a dataframe with all SWC brains used for atlases
# ------------------------------------------------------

path_of_this_script = Path(__file__).resolve()
source_csv_dir = path_of_this_script.parent.parent.parent / "data"
source_csv_path = source_csv_dir / "SWC_brain-list_for-atlases_2024-04-15.csv"
df = pd.read_csv(source_csv_path)

# Strip trailing space from str columns and from column names
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.columns = df.columns.str.strip()
# Select only data for the species of interest
species_common_name = "rat"  # as in the source csv table
df = df[df["Common name"] == species_common_name]
logger.info(f"Found {len(df)} {species_id} subjects.")

# %%
# Exract subject IDs and paths
# ----------------------------

# Rename "Specimen ID" to "subject_id"
df["subject_id"] = df["Specimen ID"]
# Assert that subject IDs are unique
assert len(df["subject_id"].unique()) == len(df), "Non-unique subject IDs"


# Use the Ceph path as the source data directory for all subjects
# (All data have been migrated there, after the table was created)
data_path_col = "Data path (raw)"  # column name in the source csv
df[data_path_col] = source_dir.as_posix()


# Find each subject's source data folder
df["subject_path"] = df.apply(
    lambda x: get_unique_folder_in_dir(
        Path(x[data_path_col]), x["subject_id"], str_position="end"
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
# We assume the standard naming conventions of SWC's serial two-photon output.

valid_colors = ["far_red", "red", "green", "blue"]  # order matters

images_list_of_dicts = []
for _, row in df.iterrows():
    sub = row["subject_id"]

    # Path to the downsampled stacks (often, but not always, in a subfolder)
    sub_path_down = row["subject_path"]
    if "downsampled_stacks" in os.listdir(sub_path_down):
        sub_path_down = sub_path_down / "downsampled_stacks"

    # Stacks are stored for each resolution in a separate folder
    # These folders are named like "010_micron", "025_micron", etc.
    stacks = [f for f in os.listdir(sub_path_down) if f.endswith("micron")]

    for stack in sorted(stacks):
        microns = int(stack.split("_")[0])
        stack_path = sub_path_down / stack

        images = [
            f
            for f in os.listdir(stack_path)
            if f.endswith(".tif") and "_ch0" in f and not f.startswith(".")
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

    print(img_dest_path)
    # if the destination path exists, skip
    if img_dest_path.exists():
        logger.debug(f"Skipping {img_dest_path} as it already exists.")
        continue

    # if image is 10 microns, symlink it (to avoid duplication of large files)
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
# Save diagnostic plots to get a quick overview of the images.
# Plots will only be generated for the low-resolution images (50um).
# Plots are saved in the rawdata/sub-<subject_id>/plots folder.

subjects = sorted([f for f in os.listdir(rawdata_dir) if f.startswith("sub-")])

for sub in tqdm(subjects):
    sub_dir = rawdata_dir / sub

    # Find all 50um images for the subject
    images = [
        f
        for f in os.listdir(sub_dir)
        if f.endswith(".tif") and "res-50um" in f
    ]

    # Plots will be saved in a subfolder
    if images:
        sub_plot_dir = sub_dir / "plots"
        sub_plot_dir.mkdir(exist_ok=True)
        logger.debug(f"Saving plots to {sub_plot_dir}...")

    for img in images:
        # load the tiff image as numpy array
        img_path = sub_dir / img
        try:
            img = load_tiff(img_path)
        except Exception as e:
            logger.error(f"Failed to load {img_path}: {e}")
            continue

        # Plot frontal (coronal) sections in a grid
        fig, _ = plot_grid(
            img,
            anat_space="PSL",
            section="frontal",
            n_slices=12,
            save_path=sub_plot_dir / f"{img_path.stem}_grid",
        )
        logger.debug(f"Saved grid plot for {img_path.stem}.")
        # Plot the image in three orthogonal views + max intensity projection
        fig, _ = plot_orthographic(
            img,
            anat_space="PSL",
            save_path=sub_plot_dir / f"{img_path.stem}_orthographic",
            mip_attenuation=0.02,
        )
        logger.debug(f"Saved orthographic plot for {img_path.stem}.")

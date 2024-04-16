"""
Built a population template for the BlackCap brain
==================================================
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
# Set up output directory and logging
# ------------------------------------

# Create the output directory
atlas_dir = Path("/nfs/nhome/live/sirmpilatzen/NIU/atlas-forge")
species_id = "BlackCap"
species_common_name = "Black cap"  # as in the source csv table
output_dir = atlas_dir / species_id
output_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
current_script_name = Path(__file__).name.split(".")[0]
logger.add(output_dir / f"{current_script_name}.log")
logger.info(f"Will save outputs to {output_dir}.")

# %%
# Load a dataframe with all SWC brains used for atlases
# ------------------------------------------------------

source_csv_dir = Path.cwd().parent / "data"
source_csv_path = source_csv_dir / "SWC_brain-list_for-atlases_2024-04-15.csv"
df = pd.read_csv(source_csv_path)

# Strip trailing space from str columns and from column names
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.columns = df.columns.str.strip()
# Select only data for the species of interest
df = df[df["Common name"] == species_common_name]
logger.info(f"Found {len(df)} {species_id} subjects.")

# %%
# Exract subject IDs and paths
# ----------------------------

df["subject_id"] = (
    df["Specimen ID"]
    .apply(lambda x: x[3:8].strip())
    .apply(lambda x: x[:-1] if x.endswith("w") else x)
)
# Assert that subject IDs are unique
assert len(df["subject_id"].unique()) == len(df), "Non-unique subject IDs"

df["Data path (raw)"] = df["Data path (raw)"].apply(
    lambda x: x.replace("nfs/ceph", "ceph")
)
df["subject_path"] = df.apply(
    lambda x: get_unique_folder_in_dir(
        Path("/" + x["Data path (raw)"]), x["subject_id"]
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
            rec_channel = ch_color in row["Imaging channel to use"]
            images_list_of_dicts.append(
                {
                    "subject_id": sub,
                    "microns": microns,
                    "channel": ch_num,
                    "color": ch_color,
                    "recommended_channel": rec_channel,
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
# Save the images (with relevant info) to the atlas directory

# Create a rawdata folder in the output directory
rawdata_dir = output_dir / "rawdata"
rawdata_dir.mkdir(exist_ok=True)

# Save dataframes to the output rawdata directory
today = date.today()

subjects_csv = rawdata_dir / f"subjects_{today}.csv"
df.to_csv(subjects_csv, index=False)
logger.info(f"Saved subject information to csv: {subjects_csv}")

images_csv = rawdata_dir / f"images_{today}.csv"
images_df.to_csv(images_csv, index=False)
logger.info(f"Saved image information to csv: {images_csv}")

# Save images to the output rawdata directory
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

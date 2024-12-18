"""Create plots showing atlas annotations overlaid on the reference image

This script needs:
- A local path to where the BrainGlobe atlas is stored
- A path to a .csv file specifying RGB color values for each region
"""
# %%
# Imports

import os
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from brainglobe_utils.IO.image import load_any
from matplotlib import pyplot as plt

from brainglobe_template_builder.plots import save_figure

# %%
# Specify paths

# Path to the BrainGlobe atlas
atlas_name = "oldenburg_blackcap_25um_v1.1"
atlas_dir = Path.home() / ".brainglobe" / atlas_name
reference_path = atlas_dir / "reference.tiff"
annotation_path = atlas_dir / "annotation.tiff"
structures_csv_path = atlas_dir / "structures.csv"

# get path of this script's parent directory
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
# Load matplotlib parameters (to allow for proper font export)
plt.style.use(current_dir / "plots.mplstyle")
# Path to the csv file containing the RGB values for each region
# Expected columns are: "acronym", "R", "G", "B"
# The .csv is in the same folder as this script
colors_csv_filename = "oldenburg_blackcap_colors.csv"
colors_csv_path = current_dir / colors_csv_filename

# Path to save the plots
save_dir = Path.home() / "Downloads"

# %%
# Load data

reference_img = load_any(reference_path)
annotation_img = load_any(annotation_path)

# Load both "structures" and "colors" csv files
structures = pd.read_csv(structures_csv_path)
colors = pd.read_csv(colors_csv_path, dtype={"R": int, "G": int, "B": int})

# Prepare structure-to-color mapping

# Merge the two dataframes on the "acronym" column
# Maintain the order of the "colors" dataframe
struct2col_df = pd.merge(
    structures, colors, on="acronym", how="right", validate="one_to_one"
)
struct2col_df.loc[:, "A"] = 0.6  # Add alpha column for transparency
for col in ["R", "G", "B"]:  # Normalise RGB values to 0-1 range
    struct2col_df[col] = round(struct2col_df[col] / 255, 3)
# Keep only necessary columns
struct2col_df = struct2col_df[["id", "acronym", "R", "G", "B", "A"]]
# Add an id=0 (empty areas), with RGBA values (0, 0, 0, 0)
new_row = pd.DataFrame(
    {"id": [0], "acronym": "empty", "R": [0], "G": [0], "B": [0], "A": [0]}
)
struct2col_df = pd.concat([new_row, struct2col_df], ignore_index=True)
# Add a mononically increasing color id column
n_colors = len(struct2col_df)
struct2col_df.loc[:, "color_id"] = range(n_colors)
struct2col_df.head(n_colors)

# %%
# Construct a new colormap using the structure-to-color mapping
# Map the monotonically increasing color_id to the RGBA values
struct_to_rgba = {
    row["color_id"]: (row["R"], row["G"], row["B"], row["A"])
    for _, row in struct2col_df.iterrows()
}
atlas_cmap = mcolors.ListedColormap(
    [struct_to_rgba[color_id] for color_id in struct_to_rgba],
    name="blackap_atlas_v1.1",
)
# Create a normalization for the colormap based on the color id values
atlas_cmap_norm = mcolors.BoundaryNorm(
    list(np.arange(n_colors + 1) - 0.5),  # list bin edges bracketing the ids
    atlas_cmap.N,  # number of colors in the colormap
)

# Remap the annotation image to the new ids
atlas_overlay = annotation_img.copy()
for id in struct2col_df["id"]:
    atlas_overlay[annotation_img == id] = struct2col_df[
        struct2col_df["id"] == id
    ]["color_id"].values[0]


# %%
# Plot the reference image with the atlas overlay

slices = [120, 260, 356]
n_slices = len(slices)
ref_slices = [reference_img.take(s, axis=0) for s in slices]

height, width = ref_slices[0].shape
fig_width = width / 100
fig_height = height / 100 * n_slices
fig, ax = plt.subplots(n_slices, 1, figsize=(fig_width, fig_height))

ann_slices = [atlas_overlay.take(s, axis=0) for s in slices]
for ann_slice in ann_slices:
    ann_slice[:, : width // 2] = 0  # Make the left half of each slice to 0

for i in range(n_slices):
    ref_frame = ref_slices[i]
    ax[i].imshow(
        ref_frame,
        cmap="gray",
        vmin=np.percentile(ref_frame, 1),
        vmax=np.percentile(ref_frame, 99),
    )

    ann_frame = ann_slices[i]
    ax[i].imshow(
        ann_frame,
        cmap=atlas_cmap,
        norm=atlas_cmap_norm,
        interpolation="nearest",
    )
    ax[i].axis("off")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
save_figure(fig, save_dir, "annotations_overlaid_on_reference")

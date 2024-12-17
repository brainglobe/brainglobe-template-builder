"""Create plots showing atlas annotations overlaid on the reference image

This script needs:
- A local path to where the BrainGlobe atlas is stored
- A path to a .csv file specifying RGB values for each region in the atlas
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

# Path to the csv file containing the RGB values for each region
# Expected columns are: "acronym", "R", "G", "B"
# The .csv is in the same folder as this script
colors_csv_filename = "oldenburg_blackcap_colors.csv"
currenct_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
colors_csv_path = currenct_script_dir / colors_csv_filename

# Path to save the plots
save_dir = Path.home() / "Downloads"

# %%
# Load data

reference_img = load_any(reference_path)
annotation_img = load_any(annotation_path)

# Load both "structures" and "colors" csv files
structures = pd.read_csv(structures_csv_path)
colors = pd.read_csv(colors_csv_path, dtype={"R": int, "G": int, "B": int})
# Merge the two dataframes on the "acronym" column
# Maintain the order of the "colors" dataframe
structures = pd.merge(
    structures, colors, on="acronym", how="right", validate="one_to_one"
)
# add an alpha column with all areas being 0.8
structures.loc[:, "A"] = 1.0
# Keep only the columns needed for the colormap
structures = structures[["id", "acronym", "R", "G", "B", "A"]]
# Divide RGB values by 255 (normalise to 0 - 1)
for col in ["R", "G", "B"]:
    structures[col] = structures[col] / 255
# Add an id=0 (empty areas), with RGBA values (0, 0, 0, 0)
new_row = pd.DataFrame(
    {"id": [0], "acronym": "empty", "R": [0], "G": [0], "B": [0], "A": [0]}
)
structures = pd.concat([new_row, structures], ignore_index=True)
# Sort daraframe by id (increasing id values)
structures = structures.sort_values(by="id")
# assign new_id in increasing order
structures.loc[:, "new_id"] = range(len(structures))
# Save the new dataframe to a csv file here
structures.to_csv(currenct_script_dir / "colors.csv", index=False)

# Create a dictionary mapping the id to the RGBA values
id_to_rgba = {
    row["new_id"]: (row["R"], row["G"], row["B"], row["A"])
    for _, row in structures.iterrows()
}
# Create a colormap using the RGBA values
annotation_cmap = mcolors.ListedColormap([id_to_rgba[id] for id in id_to_rgba])
# Create a normalization for the colormap based on the id values
annotation_cmap_norm = mcolors.BoundaryNorm(
    list(id_to_rgba.keys()), annotation_cmap.N
)

# Remap the annotation image to the new ids
for id in structures["id"].unique():
    new_id = structures.loc[structures["id"] == id, "new_id"].values[0]
    annotation_img[annotation_img == id] = new_id


# %%
# Define funciton for plotting


def plot_slices(
    reference: np.ndarray,
    slices: list[int],
    annotation=np.ndarray | None,
    axis: int = 0,
    vmin_perc: float = 1,
    vmax_perc: float = 99,
    save_path: Path | None = None,
):
    """Plot slices from a 3D image with optional annotation overlay.

    The slices are shown in a single column.

    Parameters
    ----------
    reference : np.ndarray
        Reference 3D image to plot slices from.
    slices : list[int]
        List of slice indices to plot.
    annotation : np.ndarray, optional
        Annotation image to overlay on the reference image, by default None.
        If supplied, must have the same shape as the reference image.
    axis : int, optional
        Axis along which to take slices, by default 0.
    vmin_perc : float, optional
        Lower percentile for reference image, by default 1.
    vmax_perc : float, optional
        Upper percentile for reference image, by default 99.
    save_path : Path, optional
        Path to save the plot, by default None (does not save).
    """
    n_slices = len(slices)
    ref_slices = [reference.take(s, axis=axis) for s in slices]
    height, width = ref_slices[0].shape
    fig_width = width / 100
    fig_height = height / 100 * n_slices
    fig, ax = plt.subplots(n_slices, 1, figsize=(fig_width, fig_height))

    if annotation is not None:
        ann_slices = [annotation.take(s, axis=axis) for s in slices]
        # Make the left half of each slice to 0
        for ann_slice in ann_slices:
            ann_slice[:, : width // 2] = 0

    for i in range(n_slices):
        ref_frame = ref_slices[i]
        ax[i].imshow(
            ref_frame,
            cmap="gray",
            vmin=np.percentile(ref_frame, vmin_perc),
            vmax=np.percentile(ref_frame, vmax_perc),
        )

        if annotation is not None:
            ann_frame = ann_slices[i]
            ax[i].imshow(
                ann_frame,
                cmap=annotation_cmap,
                norm=annotation_cmap_norm,
            )
        ax[i].axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    if save_path:
        save_dir, save_name = save_path.parent, save_path.name.split(".")[0]
        save_figure(fig, save_dir, save_name)


# %%
# Save plot

plot_slices(
    reference_img,
    slices=[120, 260, 356],
    annotation=annotation_img,
    save_path=save_dir / "test.png",
)

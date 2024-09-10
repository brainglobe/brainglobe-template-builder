# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt
from utils import (
    collect_coronal_slices,
    collect_template_paths,
    collect_use4template_dirs,
    load_config,
    pad_with_zeros,
    plot_slices_single_row,
    save_figure,
    setup_directories,
)

# %%
# get path of this script's parent directory
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
# Load matplotlib parameters (to allow for proper font export)
plt.style.use(current_dir / "plots.mplstyle")
# Load config file containing template building parameters
config = load_config(current_dir / "config.yaml")

# Setup directories based on config file
atlas_dir, template_dir, plots_dir = setup_directories(config)

# Load the list of transform types and number of iterations
transform_types = config["transform_types"]
n_transforms = len(transform_types)
n_iter = config["num_iterations"]
print("transform types: ", transform_types)
print("number of iterations: ", n_iter)

# Collect template images for each iteration and transform type
template_paths = collect_template_paths(template_dir, transform_types, n_iter)
# Get the path to the final template
final_stage = transform_types[-1]
final_template_path = template_paths[f"{final_stage} iter-{n_iter-1}"]

# %%
# Load the csv file containing the images used for template building
df = pd.read_csv(template_dir.parent / "use_for_template.csv")
# Collect per-subject directories containing images used for template building
use4template_dirs = collect_use4template_dirs(
    df,
    atlas_dir,
    resolution=config["resolution_um"],
    suffix=config["use4template_dir_suffix"],
)
# Paths to the asymmetric images used for template building
asym_inputs_paths = {
    subject: folder / f"{subject}_asym-brain.nii.gz"
    for subject, folder in use4template_dirs.items()
}
# add the final template to the list of images
asym_inputs_paths["final_template"] = final_template_path

# %%
# Collect coronal slices for each asymmetric input image
asym_inputs_slices = collect_coronal_slices(
    asym_inputs_paths, config["show_coronal_slice"]
)
plot_slices_single_row(
    asym_inputs_slices,
    vmin_perc=config["vmin_percentile"],
    vmax_perc=config["vmax_percentile"],
    save_path=plots_dir / "inputs_asym_brain",
)


# %%
# Plot the final average template alongside and example subject
template_img = load_nii(final_template_path, as_array=True, as_numpy=True)
template_img, pad_sizes = pad_with_zeros(template_img, target=512)

example_subject = config["example_subject"]
subject_path = asym_inputs_paths[example_subject]
subject_img = load_nii(subject_path, as_array=True, as_numpy=True)
subject_img, _ = pad_with_zeros(subject_img, target=512)

sc = AnatomicalSpace("ASR")

# %%

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axis_names = ["z", "y", "x"]
show_slices = [256, 183, 252]  # coronal, sagittal, horizontal

for i, img in enumerate([template_img, subject_img]):
    for j, (plane, labels) in enumerate(zip(sc.sections, sc.axis_labels)):
        slice_j = show_slices[j]
        ax = axs[i, j]
        ax.imshow(
            img.take(slice_j + pad_sizes[j], axis=j),
            cmap="gray",
            vmin=np.percentile(img, 1),
            vmax=np.percentile(img, 99.9),
        )
        if i == 0:
            ax.set_title(f"{plane.capitalize()} view", color="w")
            ax.text(256, 20, f"slice {slice_j}", color="w", ha="center")
        ax.set_ylabel(labels[0], color="w")
        if i == 1:
            ax.set_xlabel(labels[1], color="w")
        ax.set_xticks([])
        ax.set_yticks([])

# Make the figure background black
fig.patch.set_facecolor("black")
fig.subplots_adjust(
    left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0
)
save_figure(fig, plots_dir, "final_template_and_example_subject")


# %%

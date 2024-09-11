# %%
import os
from pathlib import Path

import pandas as pd
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt

from brainglobe_template_builder.plots import (
    collect_coronal_slices,
    collect_template_paths,
    collect_use4template_dirs,
    load_config,
    pad_with_zeros,
    plot_orthographic,
    plot_slices_single_column,
    plot_slices_single_row,
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
print("Final template path: ", final_template_path)

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

# Collect coronal slices for each asymmetric input image
asym_inputs_slices = collect_coronal_slices(
    asym_inputs_paths, config["show_slices"][0]
)
print(
    f"Collected coronal slice {config['show_slices'][0]} "
    "from each subject's asymmetric brain image"
)

# %%
plot_slices_single_row(
    asym_inputs_slices,
    vmin_perc=config["vmin_percentile"],
    vmax_perc=config["vmax_percentile"],
    save_path=plots_dir / "inputs_asym_brain_single_row",
)

# %%
plot_slices_single_column(
    asym_inputs_slices,
    vmin_perc=config["vmin_percentile"],
    vmax_perc=config["vmax_percentile"],
    save_path=plots_dir / "inputs_asym_brain_single_column",
)
print("Plotted individual subjects' asymmetric brain images")


# %%
# Plot the final average template alongside and example subject

# Find the maximum dimension size
template_img = load_nii(final_template_path, as_array=True, as_numpy=True)
target_size = max(template_img.shape)
template_img, pad_sizes = pad_with_zeros(template_img, target=target_size)

example_subject = config["example_subject"]
subject_path = asym_inputs_paths[example_subject]
subject_img = load_nii(subject_path, as_array=True, as_numpy=True)
subject_img, _ = pad_with_zeros(subject_img, target=target_size)

fig, axs = plot_orthographic(
    template_img,
    config["show_slices"],
    pad_sizes=pad_sizes,
    save_path=plots_dir / "final_template_orthographic",
)

fig, axs = plot_orthographic(
    subject_img,
    config["show_slices"],
    pad_sizes=pad_sizes,
    save_path=plots_dir / f"{example_subject}_orthographic",
)
print("Plotted final template and example subject in orthographic view")

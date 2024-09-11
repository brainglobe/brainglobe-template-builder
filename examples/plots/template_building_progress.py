"""Create plots showing the progress of template building

This script creates two plots:
1. A grid of images showing the progress across transform types and iterations.
2. An animation showing the progress across transform types and iterations.
"""
# Imports
import os
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from brainglobe_template_builder.plots import (
    collect_coronal_slices,
    collect_template_paths,
    compute_vmin_vmax_across_slices,
    load_config,
    save_figure,
    setup_directories,
)

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Collect template images for each iteration and transform type
template_paths = collect_template_paths(template_dir, transform_types, n_iter)

# Collect coronal slices for each iteration and transform type
show_coronal_slice = config["show_slices"][0]
template_slices = collect_coronal_slices(template_paths, show_coronal_slice)
print(
    f"Collected coronal slice {show_coronal_slice} for each "
    f"transform type and iteration"
)

# Calculate vmin and vmax for all slices to ensure consistent scaling
vmin, vmax = compute_vmin_vmax_across_slices(
    template_slices,
    vmin_perc=config["vmin_percentile"],
    vmax_perc=config["vmax_percentile"],
)

# Compute the aspect ratio of the 1st slice (should be same for all
width = template_slices["rigid iter-0"].shape[1]
height = template_slices["rigid iter-0"].shape[0]
aspect = width / height

# ----------------------------------------------------------------------------
# Plot all transform types and iterations in a grid
# Rows: transform types, Columns: iterations
figs, axs = plt.subplots(
    n_transforms,
    n_iter,
    figsize=(2 * aspect * n_iter, 2 * n_transforms),
)

for t, transform_type in enumerate(transform_types):
    for i in range(n_iter):
        frame = template_slices[f"{transform_type} iter-{i}"]
        ax = axs[t, i]
        ax.imshow(frame, vmin=vmin, vmax=vmax, cmap="gray")

        title = f"iter {i}" if t == 0 else ""
        ax.set_title(title, fontsize="x-large")

        ylabel = transform_type if i == 0 else ""
        ylabel = ylabel.replace("nlin", "non-linear")
        ax.set_ylabel(ylabel, fontsize="x-large")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

figs.subplots_adjust(
    left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05
)
save_figure(
    figs,
    plots_dir,
    "template_building_progress_grid",
)
print("Saved template building progress grid plot")

# ----------------------------------------------------------------------------
# Create animation of template building progress

# Create figure and axis
fig, ax = plt.subplots(figsize=(8 * aspect, 8))
ax.set_xticks([])
ax.set_yticks([])

# Initialize the plot with the first frame
frame_list = list(template_slices.values())
frame_names = list(template_slices.keys())
img = ax.imshow(frame_list[0], vmin=vmin, vmax=vmax, cmap="gray")


def update(frame_index):
    """Update function for the animation"""
    img.set_array(frame_list[frame_index])
    transform, iteration = frame_names[frame_index].split()
    transform = transform.replace("nlin", "non-linear")
    ax.set_ylabel(transform, fontsize="x-large")
    ax.set_title(iteration, fontsize="x-large")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return [img]


fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Create the animation
ani = FuncAnimation(
    fig,  # The figure object
    update,  # The update function
    frames=len(template_slices),  # Number of frames
    blit=True,  # Optimize drawing by only updating changed parts
)

# Save the animation as a gif
for fps in config["animation_fps"]:
    ani.save(
        plots_dir / f"template_building_progress_fps-{fps}.gif",
        writer="ffmpeg",
        dpi=150,
        fps=fps,
    )

print(
    "Saved template building progress gifs for fps: "
    f"{config['animation_fps']}"
)

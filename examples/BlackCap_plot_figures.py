"""Plot figures for the BlackCap paper
"""

# %%
# Imports
# -------
import os
from datetime import date
from pathlib import Path

import numpy as np
from brainglobe_utils.IO.image import load_nii
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#

# %%
# Setup logging, paths and global variables
# -----------------------------------------

# Species-specific atlas-forge directory
atlas_forge_dir = Path(
    "/Volumes/neuroinformatics/neuroinformatics/atlas-forge"
)
atlas_dir = atlas_forge_dir / "BlackCap"
assert atlas_dir.exists(), f"Atlas directory not found: {atlas_dir}"

# Set up logging
today = date.today()
current_script_name = os.path.basename(__file__).replace(".py", "")
logger.add(atlas_dir / "logs" / f"{today}_{current_script_name}.log")

# Atlas resolution in microns and mm
res_um = 25  # resolution in microns
res_str = f"res-{res_um}um"
res_mm = res_um * 1e-3  # resolution in mm (for NIfTI files)
vox_sizes = [res_mm, res_mm, res_mm]

# Define the path to the template-building directory
template_name = f"template_sym_{res_str}_n-18"
template_dir = atlas_dir / "templates" / template_name
if template_dir.exists():
    logger.info(f"Template directory: {template_dir}")
else:
    error_msg = (
        f"Template directory not found: {template_dir} ."
        f"Please check the template name {template_name}."
    )
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

# Create output directory for plots
plots_dir = template_dir / "plots"
plots_dir.mkdir(exist_ok=True)
logger.info(f"Will write outputs to plots directory: {plots_dir}")

# transform types
iterations = 4
logger.info(f"Iterations: {iterations}")
transform_types = ["rigid", "similarity", "affine", "nlin"]
folders_in_template_dir = [
    f for f in os.listdir(template_dir) if os.path.isdir(template_dir / f)
]
transform_types = [f for f in transform_types if f in folders_in_template_dir]
logger.info(f"Transform types found: {transform_types}")

# %%
# Collect template images for each iteration and transform type
# --------------------------------------------------------------

template_file_name = "template_sharpen_shapeupdate.nii.gz"

image_paths: dict = {}

for transform_type in transform_types:
    transform_type_dir = template_dir / transform_type
    image_paths[transform_type] = {}
    for iteration in range(iterations):
        transform_iteration_dir = transform_type_dir / str(iteration)
        image_paths[transform_type][f"iter-{iteration}"] = (
            transform_iteration_dir / "average" / template_file_name
        )


# %%
# Collect coronal slices for each iteration and transform type
# ------------------------------------------------------------

show_slice = 256  # 0-indexed, coronal (z) slice, axis order: z, y, x

frame_names = []
frames = []

for t, transform_type in enumerate(transform_types):
    for i in range(iterations):
        img_path = image_paths[transform_type][f"iter-{i}"]
        img = load_nii(img_path, as_array=False)
        slc = img.slicer[show_slice : show_slice + 1, :, :]
        frames.append(slc.get_fdata().squeeze())
        frame_names.append(f"{transform_type} {i}")

# %%
# Create animation
# ----------------
# Calculate vmin and vmax for all frames to ensure consistent scaling
vmin = np.min([frame.min() for frame in frames])
vmax = np.max([frame.max() for frame in frames])

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")

# Initialize the plot with the first frame
img = ax.imshow(frames[0], vmin=vmin, vmax=vmax, cmap="gray")


def update(frame_index):
    """Update function for the animation"""
    img.set_array(frames[frame_index])
    ax.set_title(frame_names[frame_index], fontsize="x-large")
    return [img]


fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Create the animation
ani = FuncAnimation(
    fig,  # The figure object
    update,  # The update function
    frames=len(frames),  # Number of frames
    interval=200,  # Interval between frames in milliseconds
    blit=True,  # Use blitting for better performance
)

# Save the animation as a gif
for fps in [2, 4, 8]:
    ani.save(
        plots_dir / f"transforms_iterations_animation_fps-{fps}.gif",
        writer="ffmpeg",
        dpi=300,
        fps=fps,
    )

# %%
# Plot only the last iteration for each transform type
# ----------------------------------------------------

# Create figure and axis
fig, axs = plt.subplots(1, len(transform_types), figsize=(14, 3))
for t, transform_type in enumerate(transform_types):
    frame = frames[(t + 1) * iterations - 1]
    ax = axs[t]
    ax.imshow(frame, vmin=vmin, vmax=vmax, cmap="gray")
    ax.axis("off")
    ax.set_title(transform_type, fontsize="x-large")
fig.subplots_adjust(
    left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05
)

# Save the figure
fig.savefig(plots_dir / "template_across_transform_types.png", dpi=300)

# %%
# Plot all iterations of the nlin transform type only
# ---------------------------------------------------

n_transform_types = len(transform_types)
# Create figure and axis
fig, axs = plt.subplots(1, iterations, figsize=(14, 3))
for i in range(iterations):
    frame = frames[iterations * (n_transform_types - 1) + i]
    ax = axs[i]
    ax.imshow(frame, vmin=vmin, vmax=vmax, cmap="gray")
    ax.axis("off")
    ax.set_title(f"nlin iter {i}", fontsize="x-large")
fig.subplots_adjust(
    left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05
)
fig.savefig(plots_dir / "template_across_nlin_iterations.png", dpi=300)

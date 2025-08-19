"""Quantify sharpness of template sand individual images
========================================================

Our chosen sharpness metric is edge SNR, i.e. the ratio of
the mean gradient magnitude on edges
to the standard deviation of the gradient magnitude on non-edges.

Gradient magnitude is computed with sigma = 1.
Edge masks are created by a sequence of filters:
- Gaussian smoothing (sigma = 2)
- Sobel filter
- Otsu's thresholding
- Binary dilation (radius = 1)
"""

# %%
# Imports
# -------

import os
from pathlib import Path

import numpy as np
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt

from brainglobe_template_builder.metrics import (
    create_edge_mask_3d,
    edge_snr_3d,
    gradient_magnitude,
)
from brainglobe_template_builder.plots import (
    collect_template_paths,
    load_config,
    setup_directories,
)

# %%
# get path of this script's parent directory
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
# Load matplotlib parameters (to allow for proper font export)
plt.style.use(current_dir / "plots.mplstyle")
# Load config file containing template building parameters
config = load_config(current_dir / "config_25um.yaml")

# Setup directories based on config file
atlas_dir, template_dir, plots_dir = setup_directories(config)

# Load the list of transform types and number of iterations
transform_types = config["transform_types"]
n_transforms = len(transform_types)
n_iter = config["num_iterations"]
final_stage = transform_types[-1]
final_iter = f"iter-{n_iter-1}"
print("transform types: ", transform_types)
print("number of iterations: ", n_iter)

# Collect template images for each iteration and transform type
template_paths = collect_template_paths(template_dir, transform_types, n_iter)
# Get paths to the final iteration of each transform stage
template_paths_final_iter = {
    k: v for k, v in template_paths.items() if k.endswith(final_iter)
}
# Get the path to the very final template (final stage, final iteration)
final_template_path = template_paths[f"{final_stage} {final_iter}"]
print("Final template path: ", final_template_path)

# Get paths to individual sample brain images
with open(template_dir / "brain_paths.txt", "r") as f:
    sample_paths = [
        Path(line.strip().replace("/ceph/neuroinformatics", "/media/ceph-niu"))
        for line in f.readlines()
    ]

# %%

template_images: dict[str, np.ndarray] = {}
gradient_images: dict[str, np.ndarray] = {}
edge_masks: dict[str, np.ndarray] = {}

for label, path in template_paths_final_iter.items():
    template_images[label] = load_nii(path, as_array=True, as_numpy=True)
    gradient_images[label] = gradient_magnitude(
        template_images[label], sigma=1.0
    )
    edge_masks[label] = create_edge_mask_3d(
        template_images[label], sigma=2.0, dilate_radius=1
    )
    print(f"Loaded and processed {label}.")

# %%
# Compute vmax values for comparable plots
vmax_img = np.max([np.percentile(v, 99) for v in template_images.values()])
vmax_grad = np.max([np.percentile(v, 99) for v in gradient_images.values()])

# %%

fig, axes = plt.subplots(4, 3, figsize=(8, 8), sharex=True, sharey=True)

slice_idx = config["show_slices"][0]

for row, label in enumerate(template_paths_final_iter.keys()):
    stage = label.split()[0]

    axes[row, 0].imshow(
        template_images[label][slice_idx, :, :],
        cmap="Greys_r",
        vmin=0,
        vmax=vmax_img,
    )
    axes[row, 0].set_title(stage)

    axes[row, 1].imshow(
        gradient_images[label][slice_idx, :, :],
        cmap="Greys_r",
        vmin=0,
        vmax=vmax_grad,
    )
    axes[row, 1].set_title("Gradient magnitude")

    axes[row, 2].imshow(
        edge_masks[label][slice_idx, :, :],
        cmap="binary_r",
    )
    axes[row, 2].set_title("Edge Mask")

    for c in range(3):
        axes[row, c].axis("off")


# %%

edge_snrs: dict[str, float] = {}

for label in template_images.keys():
    stage = label.split()[0]
    edge_snr = edge_snr_3d(gradient_images[label], edge_masks[label])
    edge_snrs[stage] = edge_snr
    print(f"Computed edge SNR for {stage}: {edge_snr}")


# %%

sample_edge_snrs: dict[str, float] = {}

for sample_path in sample_paths:
    sample_path = sample_path
    sample_name = sample_path.stem
    print(f"Processing sample: {sample_name}")

    sample_img = load_nii(sample_path, as_array=True, as_numpy=True)

    gmag = gradient_magnitude(sample_img, sigma=1.0)
    edge_mask = create_edge_mask_3d(sample_img, sigma=2.0, dilate_radius=1)
    edge_snr = edge_snr_3d(gmag, edge_mask)
    print(f"Computed edge SNR for {sample_name}: {edge_snr}")
    sample_edge_snrs[sample_name] = edge_snr


# %%

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Plot a boxplot of all sample_edge_snrs
axes[0].set_title("Individual Samples")
axes[0].scatter(
    [1] * len(sample_edge_snrs),
    list(sample_edge_snrs.values()),
    facecolors="white",
    edgecolors="blue",
    alpha=0.7,
    s=30,
    lw=2,
)
axes[0].set_xticklabels(["individual samples"])
axes[0].set_ylabel("Edge SNR")

# Plot a line plot of edge SNRs by template stage
axes[1].set_title("Template Stages")
axes[1].plot(
    list(edge_snrs.keys()), list(edge_snrs.values()), marker="o", color="blue"
)
axes[1].set_xlabel("Template Stage")
axes[1].set_ylabel("Edge SNR")

# %%

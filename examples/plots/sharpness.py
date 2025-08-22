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
template_paths = collect_template_paths(
    template_dir,
    transform_types,
    n_iter,
    template_file_name="template_sharpen_shapeupdate.nii.gz",
)
# Collect template brain masks for each iteration and transform type
template_mask_paths = collect_template_paths(
    template_dir,
    transform_types,
    n_iter,
    template_file_name="mask_shapeupdate.nii.gz",
)
# Keep only paths to the final iteration of each transform stage
for path_dict in [template_paths, template_mask_paths]:
    for key in list(path_dict.keys()):
        if not key.endswith(final_iter):
            del path_dict[key]

print("\nWill use the following templates images:\n")
for key in template_paths.keys():
    folder = template_paths[key].parent
    template_img = template_paths[key].name
    template_mask = template_mask_paths[key].name
    print(f"stage-{key}:")
    print(f"  directory: {folder}")
    print(f"  template image: {template_img}")
    print(f"  mask image: {template_mask}")


# Get paths to individual sample brain images and masks
with open(template_dir / "brain_paths.txt", "r") as f:
    input_paths = [
        Path(line.strip().replace("/ceph/neuroinformatics", "/media/ceph-niu"))
        for line in f.readlines()
    ]
    input_dirs = sorted(list(set(p.parent for p in input_paths)))
    subjects = [d.parent.name for d in input_dirs]
    brain_paths = {
        sub: d / f"{sub}_asym-brain.nii.gz"
        for sub, d in zip(subjects, input_dirs)
    }
    mask_paths = {
        sub: d / f"{sub}_asym-mask.nii.gz"
        for sub, d in zip(subjects, input_dirs)
    }
    print(f"\nFound images for {len(brain_paths)} individual samples:\n")

for sub in subjects:
    print(f"Subject: {sub}")
    print(f"  directory: {brain_paths[sub].parent}")
    print(f"  brain: {brain_paths[sub].name}")
    print(f"  mask: {mask_paths[sub].name}")

# %%
# Load images from the final iteration of each template-building stage,
# as well as indivual subject images, and compute their
# gradient magnitude, edge/non-edge masks, and edge SNR.

combined_img_paths = {**template_paths, **brain_paths}
combined_mask_paths = {**template_mask_paths, **mask_paths}

results: dict[str, dict[str, np.ndarray]] = {}

for label, img_path in combined_img_paths.items():
    print(f"Processing {label}...")
    img = load_nii(img_path, as_array=True, as_numpy=True)
    gmag = gradient_magnitude(img, sigma=1.0)
    brain_mask = load_nii(
        combined_mask_paths[label], as_array=True, as_numpy=True
    ).astype(bool)
    edge_mask = create_edge_mask_3d(img, sigma=2.0, dilate_radius=1)
    non_edge_mask = np.logical_and(brain_mask, ~edge_mask)
    edge_snr = edge_snr_3d(gmag, edge_mask, non_edge_mask)

    results[label] = {
        "image": img,
        "gradient": gmag,
        "edge_mask": edge_mask,
        "non_edge_mask": non_edge_mask,
        "edge_snr": edge_snr,
    }
    print(f"Computed edge SNR for {label}: {edge_snr:.2f}")

# Load the individual sample images
# and compute their gradient magnitudes, edge/non-edge masks, and edge SNR.

sample_results: dict[str, dict[str, np.ndarray]] = {}

for sub, brain_path in brain_paths.items():
    mask_path = mask_paths[sub]

# %%
# Plot slices

# Compute vmax values for comparable plots
# vmax_img = np.max([np.percentile(v, 99) for v in template_images.values()])
# vmax_grad = np.max(
#     [np.percentile(v, 99) for v in template_gradients.values()]
# )

# fig, axes = plt.subplots(4, 3, figsize=(8, 8), sharex=True, sharey=True)

# slice_idx = config["show_slices"][0]

# for row, label in enumerate(template_paths_final_iter.keys()):
#     stage = label.split()[0]

#     axes[row, 0].imshow(
#         template_images[label][slice_idx, :, :],
#         cmap="Greys_r",
#         vmin=0,
#         vmax=vmax_img,
#     )
#     axes[row, 0].set_title(stage)

#     axes[row, 1].imshow(
#         template_gradients[label][slice_idx, :, :],
#         cmap="Greys_r",
#         vmin=0,
#         vmax=vmax_grad,
#     )
#     axes[row, 1].set_title("Gradient magnitude")

#     axes[row, 2].imshow(
#         template_edge_masks[label][slice_idx, :, :],
#         cmap="binary_r",
#     )
#     axes[row, 2].set_title("Edge Mask")

#     for c in range(3):
#         axes[row, c].axis("off")


# %%

# fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# # Plot a boxplot of all sample_edge_snrs
# axes[0].set_title("Individual Samples")
# axes[0].scatter(
#     [1] * len(sample_edge_snrs),
#     list(sample_edge_snrs.values()),
#     facecolors="white",
#     edgecolors="blue",
#     alpha=0.7,
#     s=30,
#     lw=2,
# )
# axes[0].set_xticklabels(["individual samples"])
# axes[0].set_ylabel("Edge SNR")

# # Plot a line plot of edge SNRs by template stage
# axes[1].set_title("Template Stages")
# axes[1].plot(
#     list(edge_snrs.keys()),
#     list(edge_snrs.values()),
#     marker="o",
#     color="blue"
# )
# axes[1].set_xlabel("Template Stage")
# axes[1].set_ylabel("Edge SNR")

# %%

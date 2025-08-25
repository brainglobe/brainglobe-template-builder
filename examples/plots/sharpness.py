"""Quantify sharpness of template sand individual images
========================================================

Our chosen sharpness metric is edge SNR, i.e. the ratio of
the mean gradient magnitude on edges
to the standard deviation of the gradient magnitude on non-edges.

Gradient magnitude is computed with sigma = 1.

Edge masks are created by a sequence of filters:
- Gaussian smoothing with sigma = 2 (0.5 mm)
- Sobel filter
- Otsu's thresholding
- Binary dilation with radius = 4 voxels (1 mm)

"""

# %%
# Imports
# -------

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt

from brainglobe_template_builder.metrics import (
    create_edge_mask_3d,
    edge_snr_3d,
    gradient_magnitude,
)
from brainglobe_template_builder.plots import (
    collect_template_paths,
    compute_vmin_vmax_across_slices,
    load_config,
    save_figure,
    setup_directories,
)

# %%
# Specify paths and plot configuration
# ------------------------------------
# Paths are collected from various stages of the template building process.
# Additionally, paths to individual subject images (that went into building
# the template) are also collected.

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
template_img_paths = collect_template_paths(
    template_dir,
    transform_types,
    n_iter,
    template_file_name="template_sharpen_shapeupdate.nii.gz",
)

# Keep only paths to the final iteration of each transform stage
# for key in list(template_img_paths.keys()):
#     if not key.endswith(final_iter):
#         del template_img_paths[key]

print("\nWill use the following templates images:\n")
for key in template_img_paths.keys():
    folder = template_img_paths[key].parent
    template_img = template_img_paths[key].name
    print(f"stage-{key}:")
    print(f"  directory: {folder}")
    print(f"  template image: {template_img}")

# Get paths to individual sample brain images and masks
with open(template_dir / "brain_paths.txt", "r") as f:
    input_paths = [
        Path(line.strip().replace("/ceph/neuroinformatics", "/media/ceph-niu"))
        for line in f.readlines()
    ]
    input_dirs = sorted(list(set(p.parent for p in input_paths)))
    subjects = [d.parent.name for d in input_dirs]
    sample_img_paths = {
        sub: d / f"{sub}_asym-brain.nii.gz"
        for sub, d in zip(subjects, input_dirs)
    }
    n_samples = len(sample_img_paths)
    print(f"\nFound images for {n_samples} individual samples:\n")

for sub in subjects:
    print(f"Subject: {sub}")
    print(f"  directory: {sample_img_paths[sub].parent}")
    print(f"  brain: {sample_img_paths[sub].name}")

# %%
# Analyse images
# --------------
# Load images from the final iteration of each template-building stage,
# as well as indivual subject images, and compute their
# gradient magnitude, edge/non-edge masks, and edge SNR.

combined_img_paths = {**template_img_paths, **sample_img_paths}

results: dict[str, dict[str, np.ndarray]] = {}

for label, img_path in combined_img_paths.items():
    print(f"Processing {label}...")
    img = load_nii(img_path, as_array=True, as_numpy=True)
    gmag = gradient_magnitude(img, sigma=1.0)
    edge_mask = create_edge_mask_3d(img, sigma=2.0, dilate_radius=4)
    edge_snr = edge_snr_3d(gmag, edge_mask)

    results[label] = {
        "image": img,
        "gradient": gmag,
        "edge_mask": edge_mask,
        "edge_snr": edge_snr,
    }
    print(f"Computed edge SNR for {label}: {edge_snr:.2f}")

# Put the edge SNR results into a DataFrame
results_df = pd.DataFrame()
results_df["label"] = list(results.keys())
results_df["category"] = "template"
results_df.loc[
    results_df["label"].str.startswith("sub-"), "category"
] = "individual"
results_df["edge_snr"] = [v["edge_snr"] for v in results.values()]


# %%
# Plot images
# -----------


def plot_gradients_edges(
    image_dict: dict[str, dict[str, np.ndarray]],
    slice_idx: int | None = None,
    sharev: bool = False,
    save_path: Path | None = None,
):
    """Plot gradient magnitudes and edge masks for each image.

    Parameters
    ----------
    image_dict: dict[str, dict[str, np.ndarray]]
        A dictionary mapping image labels (subject ID or template stage) to
        a dictionary containing the following keys:
        - "image": The original image data.
        - "gradient": The gradient magnitude of the image.
        - "edge_mask": The edge mask of the image.
        - "edge_snr": The edge SNR value (float).
    slice_idx: int | None
        The index of the coronal slice to plot.
        If None (default), show the mid-coronal slice.
    sharev: bool
        Whether to share vmin, vmax across subplots.
    save_path: Path | None
        The path to save the figure.
        If None (default), do not save the figure.

    """
    if slice_idx is None:
        slice_idx = list(image_dict.values())[0]["image"].shape[0] // 2

    n_rows = len(image_dict)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(8, n_rows * 2), sharex=True, sharey=True
    )

    for row, label in enumerate(image_dict.keys()):
        img, grad, edge_mask, snr = (
            image_dict[label]["image"][slice_idx, :, :],
            image_dict[label]["gradient"][slice_idx, :, :],
            image_dict[label]["edge_mask"][slice_idx, :, :],
            image_dict[label]["edge_snr"],
        )

        if sharev:
            vmin_img, vmax_img = compute_vmin_vmax_across_slices(
                {k: v["image"][slice_idx, :, :] for k, v in image_dict.items()}
            )
            vmin_grad, vmax_grad = compute_vmin_vmax_across_slices(
                {
                    k: v["gradient"][slice_idx, :, :]
                    for k, v in image_dict.items()
                }
            )
        else:
            vmin_img, vmax_img = np.percentile(img, [1, 99])
            vmin_grad, vmax_grad = np.percentile(grad, [1, 99])

        axes[row, 0].imshow(img, cmap="Greys_r", vmin=vmin_img, vmax=vmax_img)

        axes[row, 1].imshow(
            grad,
            cmap="Greys_r",
            vmin=vmin_grad,
            vmax=vmax_grad,
        )

        axes[row, 2].imshow(
            grad, cmap="Greys_r", vmin=vmin_grad, vmax=vmax_grad
        )
        overlay = np.ma.masked_where(~edge_mask, edge_mask)
        axes[row, 2].imshow(
            overlay,
            cmap="Reds_r",
            alpha=0.8,
            interpolation="nearest",
        )
        axes[row, 2].text(
            0.5,
            0.05,
            f"Edge SNR = {snr:.2f}",
            color="white",
            ha="center",
            va="center",
            transform=axes[row, 2].transAxes,
        )

        if row == 0:
            axes[row, 0].set_title("Image")
            axes[row, 1].set_title("Gradient magnitude")
            axes[row, 2].set_title("Edge mask")

        for c in range(3):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

        axes[row, 0].set_ylabel(label)

    plt.tight_layout()

    if save_path:
        save_dir, save_name = save_path.parent, save_path.name.split(".")[0]
        save_figure(fig, save_dir, save_name)


# %%

plot_gradients_edges(
    {
        "Rigid": results["rigid iter-3"],
        "Similarity": results["similarity iter-3"],
        "Affine": results["affine iter-3"],
        "Non-linear": results["nlin iter-3"],
    },
    slice_idx=config["show_slices"][0],
    sharev=True,
    save_path=plots_dir / "templates_gradients_edges.png",
)

# %%

plot_gradients_edges(
    {k: v for k, v in results.items() if k.startswith("sub-")},
    slice_idx=config["show_slices"][0],
    sharev=False,
    save_path=plots_dir / "subjects_gradients_edges.png",
)


# %%
example_sub = config["example_subjects"][0]

plot_gradients_edges(
    {
        "Individual Sample": results[example_sub],
        "Average Template": results["nlin iter-3"],
    },
    slice_idx=config["show_slices"][0],
    sharev=False,
    save_path=plots_dir / "sub-BC15_vs_template_gradients_edges.png",
)


# %%

# Massage df to prepare for plotting
sample_df = results_df[results_df["category"] == "individual"].copy()
sample_df["stage"] = sample_df["label"].copy()
sample_df["iter"] = "N/A"

templates_df = results_df[results_df["category"] == "template"].copy()
templates_df["stage"] = templates_df["label"].apply(lambda x: x.split(" ")[0])
templates_df["stage"] = templates_df["stage"].replace("nlin", "non-linear")
templates_df["iter"] = templates_df["label"].apply(lambda x: x.split("-")[1])
templates_final_iter_df = templates_df[templates_df["iter"] == str(n_iter - 1)]

plot_df = pd.concat([sample_df, templates_final_iter_df], axis=0).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.stripplot(
    plot_df,
    y="label",
    x="edge_snr",
    ax=ax,
    jitter=False,
    color="0.5",
    size=8,
    legend=False,
)
ax.set_xlim(3.5, 6.5)
ax.set_xticks(np.arange(3.5, 6.6, 1))

ax.axhline(n_samples - 0.5, color="0.1", linestyle="--")
ax.text(
    6.4, 0, "Individual Samples", ha="right", va="center", fontsize="large"
)
ax.text(6.4, n_samples, "Templates", ha="right", va="center", fontsize="large")

ax.set_ylabel(" ")
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df.stage.values)
ax.set_title("Edge SNR (a.u.)")
ax.set_xlabel("")

plt.tight_layout()

save_figure(fig, plots_dir, "edge_snr_comparison")

plot_df.to_csv(plots_dir / "edge_snr_comparison.csv", index=False)
# %%

"""Quantify sharpness of template sand individual images
========================================================

Our chosen sharpness metric is edge SNR, i.e. the ratio of
the mean gradient magnitude on edges
to the mean gradient magnitude on non-edges.

Gradient magnitude is computed with sigma = 1.

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
    gradient_magnitude,
)
from brainglobe_template_builder.plots import (
    collect_template_paths,
    compute_vmin_vmax_across_slices,
    linear_rescale,
    load_config,
    pad_with_zeros,
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

# Path to save the plots (overrides plots_dir)
niu_dropbox_dir = Path.home() / "Dropbox" / "NIU"
figures_dir = niu_dropbox_dir / "resources" / "figures"
plots_dir = figures_dir / "BlackCap_Atlas_Male" / "img"

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
template_mask_paths = collect_template_paths(
    template_dir, transform_types, n_iter, "mask_shapeupdate.nii.gz"
)

# Keep only paths to the final iteration of each transform stage
for key in list(template_img_paths.keys()):
    if not key.endswith(final_iter):
        del template_img_paths[key]

print("\nWill use the following templates images:\n")
for key in template_img_paths.keys():
    folder = template_img_paths[key].parent
    template_img = template_img_paths[key].name
    template_mask = template_mask_paths[key].name
    print(f"stage-{key}:")
    print(f"  directory: {folder}")
    print(f"  template image: {template_img}")
    print(f"  template mask: {template_mask}")

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
    sample_mask_paths = {
        sub: d / f"{sub}_asym-mask.nii.gz"
        for sub, d in zip(subjects, input_dirs)
    }
    n_samples = len(sample_img_paths)
    print(f"\nFound images for {n_samples} individual samples:\n")

for sub in subjects:
    print(f"Subject: {sub}")
    print(f"  directory: {sample_img_paths[sub].parent}")
    print(f"  brain: {sample_img_paths[sub].name}")
    print(f"  mask: {sample_mask_paths[sub].name}")

# %%
# Load images
# -----------
# Load images from the final iteration of each template-building stage,
# as well as indivual subject images.

combined_img_paths = {**template_img_paths, **sample_img_paths}
combined_mask_paths = {**template_mask_paths, **sample_mask_paths}

combined_images: dict[str, np.ndarray] = {
    label: load_nii(combined_img_paths[label], as_array=True, as_numpy=True)
    for label in combined_img_paths
}
combined_masks: dict[str, np.ndarray] = {
    label: load_nii(combined_mask_paths[label], as_array=True, as_numpy=True)
    for label in combined_mask_paths
}

final_template = load_nii(
    template_img_paths[f"{final_stage} {final_iter}"],
    as_array=True,
    as_numpy=True,
)
final_mask = load_nii(
    template_mask_paths[f"{final_stage} {final_iter}"],
    as_array=True,
    as_numpy=True,
)

# %%
# Analyze images
# --------------
# Compute gradient magnitude, edge/non-edge masks, and edge SNR.

results: dict[str, dict[str, np.ndarray]] = {}

for label, img_path in combined_img_paths.items():
    img = combined_images[label]
    img_mask = combined_masks[label]
    img = linear_rescale(
        source_image=img,
        target_image=final_template,
        source_mask=img_mask,
        target_mask=final_mask,
    )
    gmag = gradient_magnitude(img, sigma=1.0)
    edge_mask = create_edge_mask_3d(img, sigma=2.0, dilate_radius=4)
    edge_mean_grad = gmag[edge_mask].mean()
    non_edge_mean_grad = gmag[~edge_mask].mean()
    edge_snr = edge_mean_grad / (non_edge_mean_grad + 1e-12)

    results[label] = {
        "norm_image": img,
        "gradient": gmag,
        "edge_mask": edge_mask,
        "edge_mean_grad": edge_mean_grad,
        "non_edge_mean_grad": non_edge_mean_grad,
        "edge_snr": edge_snr,
    }
    print(
        f"{label}:\n"
        f"  Edge mean grad: {edge_mean_grad:.4f}\n"
        f"  Non-edge mean grad: {non_edge_mean_grad:.4f}\n"
        f"  Edge SNR: {edge_snr:.4f}"
    )


# Put the edge SNR results into a DataFrame
results_df = pd.DataFrame()
results_df["label"] = list(results.keys())
results_df["category"] = "template"
results_df.loc[
    results_df["label"].str.startswith("sub-"), "category"
] = "individual"
results_df["edge_mean_grad"] = [v["edge_mean_grad"] for v in results.values()]
results_df["non_edge_mean_grad"] = [
    v["non_edge_mean_grad"] for v in results.values()
]
results_df["edge_snr"] = [v["edge_snr"] for v in results.values()]


# %%
# Plot a histrogram for each gradient image

template_gradients = {
    "Rigid": results["rigid iter-3"]["gradient"],
    "Similarity": results["similarity iter-3"]["gradient"],
    "Affine": results["affine iter-3"]["gradient"],
    "Non-linear": results["nlin iter-3"]["gradient"],
}

subject_gradients = {
    sub: results[sub]["gradient"] for sub in results if sub.startswith("sub-")
}

# Make y axis log scale
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4),
    sharey=True,
    sharex=True,
    subplot_kw={"yscale": "log"},
)

bins = np.linspace(0, 18, 50)
for label, grad in template_gradients.items():
    axes[0].hist(
        grad.ravel(),
        bins=bins,
        histtype="step",
        label=label,
    )
    axes[0].set_title("Average Templates")
    axes[0].set_xlabel("Gradient magnitude")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()


for label, grad in subject_gradients.items():
    axes[1].hist(
        grad.ravel(),
        bins=bins,
        histtype="step",
        label=label,
    )
    axes[1].set_title("Individual Samples")
    axes[1].set_xlabel("Gradient magnitude")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

plt.suptitle("Gradient magnitude histograms")
plt.tight_layout()
plt.savefig(plots_dir / "gradient_magnitude_histograms.png")


# %%
# Plot images
# -----------


def plot_gradients_edges(
    image_dict: dict[str, dict[str, np.ndarray]],
    slice_idx: int | None = None,
    pad: bool = False,
    save_path: Path | None = None,
):
    """Plot gradient magnitudes and edge masks for each image.

    Parameters
    ----------
    image_dict: dict[str, dict[str, np.ndarray]]
        A dictionary mapping image labels (subject ID or template stage) to
        a dictionary containing the following keys:
        - "norm_image": The image data rescaled to match the final template.
        - "gradient": The gradient magnitude of the image.
        - "edge_mask": The edge mask of the image.
        - "edge_snr": The edge SNR value (float).
    slice_idx: int | None
        The index of the coronal slice to plot.
        If None (default), show the mid-coronal slice.
    pad: bool
        Whether to pad the images to a square aspect ratio.
    save_path: Path | None
        The path to save the figure.
        If None (default), do not save the figure.

    """
    if slice_idx is None:
        slice_idx = list(image_dict.values())[0]["norm_image"].shape[0] // 2

    n_rows = len(image_dict)
    fig_width = 6 if pad else 8
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(fig_width, n_rows * 2), sharex=True, sharey=True
    )

    vmin_img, vmax_img = compute_vmin_vmax_across_slices(
        {k: v["norm_image"][slice_idx, :, :] for k, v in image_dict.items()}
    )
    vmin_grad, vmax_grad = compute_vmin_vmax_across_slices(
        {k: v["gradient"][slice_idx, :, :] for k, v in image_dict.items()}
    )

    for row, label in enumerate(image_dict.keys()):
        img, grad, edge_mask, snr = (
            image_dict[label]["norm_image"][slice_idx, :, :],
            image_dict[label]["gradient"][slice_idx, :, :],
            image_dict[label]["edge_mask"][slice_idx, :, :],
            image_dict[label]["edge_snr"],
        )

        if pad:
            target_size = max(img.shape)
            img, grad, edge_mask = map(
                lambda x: pad_with_zeros(x, target=target_size)[0],
                [img, grad, edge_mask],
            )

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
            0.1,
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
    pad=False,
    save_path=plots_dir / "templates_gradients_edges.png",
)

# %%

plot_gradients_edges(
    {k: v for k, v in results.items() if k.startswith("sub-")},
    slice_idx=config["show_slices"][0],
    pad=False,
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
    pad=True,
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

# %%

fig, ax = plt.subplots(1, 1, figsize=(3.5, 4))

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
ax.set_xlim(5, 11)
ax.set_xticks(np.arange(5, 11.1, 2))

ax.axhline(n_samples - 0.5, color="0.1", linestyle="--")
ax.set_ylabel(" ")
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df.stage.values)
ax.set_title("Edge SNR (a.u.)")
ax.set_xlabel("")

plt.tight_layout()

save_figure(fig, plots_dir, "edge_snr_comparison")

plot_df.to_csv(plots_dir / "edge_snr_comparison.csv", index=False)

# %%

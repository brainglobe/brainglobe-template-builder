"""Quantify sharpness of template and individual images
=======================================================

We consider the following sharpness metrics:
- Variance of Laplacian
- Gaussian gradient magnitudes at different sigmas
"""

# %%
# Imports
# -------

import os
from pathlib import Path

import numpy as np
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt

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
print("transform types: ", transform_types)
print("number of iterations: ", n_iter)

# Collect template images for each iteration and transform type
template_paths = collect_template_paths(template_dir, transform_types, n_iter)

# Get the path to the final template
final_stage = transform_types[-1]
final_template_path = template_paths[f"{final_stage} iter-{n_iter-1}"]
print("Final template path: ", final_template_path)


# %%


def get_variance_of_laplacian(image: np.ndarray) -> float:
    """Calculate the variance of the Laplacian of a 3D image."""
    from scipy.ndimage import laplace

    laplacian = laplace(image, mode="constant")
    return np.var(laplacian)


def get_gradient_magnitude(image: np.ndarray, sigma=0) -> np.ndarray:
    """Calculate the gradient magnitude of a 3D image."""
    from scipy.ndimage import gaussian_gradient_magnitude

    # Use a small sigma to avoid smoothing too much
    gradient = gaussian_gradient_magnitude(image, sigma=sigma, mode="constant")
    return gradient


# %%
# Load each template image and calculate the variance of Laplacian


template_images: dict[str, np.ndarray] = {}
var_of_laplacian: dict[str, float] = {}

for stage, path in template_paths.items():
    print(f"Loading template image for {stage}...")
    template_images[stage] = load_nii(path, as_array=True, as_numpy=True)
    var_of_laplacian[stage] = get_variance_of_laplacian(template_images[stage])
    print(f"Variance of Laplacian for {stage}: {var_of_laplacian[stage]:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(
    list(var_of_laplacian.keys()), list(var_of_laplacian.values()), marker="o"
)
plt.xticks(rotation=45)
plt.title("Variance of Laplacian for Template Stages")


# %%
# Plot histograms of gradient magnitudes for each template stage
# and for different Gaussian smoothing sigmas.
# We only look at the last iteration of each stage.

gradient_magnitudes: dict[str, np.ndarray] = {}

fig, axs = plt.subplots(1, 4, figsize=(12, 4))

stages = ["rigid", "similarity", "affine", "nlin"]
iters = [0, 1, 2, 3]
cmap = plt.get_cmap("viridis", len(stages))

for s, sigma in enumerate([0, 1, 3, 5]):
    for i, stage in enumerate(stages):
        last_iter = f"{stage} iter-{iters[-1]}"
        ax = axs[s]
        grad_amplitude = get_gradient_magnitude(
            template_images[last_iter], sigma=sigma
        )
        max_grad = np.ceil(np.max(grad_amplitude))
        ax.hist(
            grad_amplitude.flatten(),
            density=True,
            lw=1.5,
            bins=np.linspace(0, max_grad, 50),
            histtype="step",
            label=last_iter,
            color=cmap(i),
        )
        ax.set_yscale("log")
        ax.set_xlabel("Gaussian Gradient Magnitude")
        ax.set_ylabel("Density")
        ax.set_title(f"Sigma = {sigma}")
        ax.legend()

        # Store the gradient magnitude for later visualization
        gradient_magnitudes[f"{last_iter} sigma-{sigma}"] = grad_amplitude


# %%

fig, axs = plt.subplots(4, 4, figsize=(12, 8), sharex=True, sharey=True)

max_grads = [10, 1.5, 0.4, 0.3]


for i, stage in enumerate(gradient_magnitudes.keys()):
    ax = axs[i // 4, i % 4]
    ax.imshow(
        gradient_magnitudes[stage][256, :, :],
        vmin=0,
        vmax=max_grads[i // 4],
        cmap="viridis",
    )
    ax.set_title(stage)
    ax.axis("off")

# %%

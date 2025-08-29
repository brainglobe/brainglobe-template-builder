# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt

from brainglobe_template_builder.io import save_nii
from brainglobe_template_builder.metrics import gradient_magnitude
from brainglobe_template_builder.plots import (
    collect_coronal_slices,
    collect_template_paths,
    collect_use4template_dirs,
    linear_rescale,
    load_config,
    pad_with_zeros,
    plot_inset_comparison,
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
print("transform types: ", transform_types)
print("number of iterations: ", n_iter)

# Collect template images for each iteration and transform type
template_paths = collect_template_paths(
    template_dir,
    transform_types,
    n_iter,
)
template_mask_paths = collect_template_paths(
    template_dir, transform_types, n_iter, "mask_shapeupdate.nii.gz"
)
# Get the path to the final template
final_stage = transform_types[-1]
final_template_path = template_paths[f"{final_stage} iter-{n_iter-1}"]
final_mask_path = template_mask_paths[f"{final_stage} iter-{n_iter-1}"]
print("Final template path: ", final_template_path)
print("Final mask path: ", final_mask_path)

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
# Collect unique subject names used for this template
brain_paths_txt = template_dir / "brain_paths.txt"
brain_paths = np.loadtxt(brain_paths_txt, dtype=str).tolist()
used_subj_names = {Path(path).name.split("_")[0] for path in brain_paths}
# Paths to the asymmetric images used for this template
asym_inputs_paths = {
    subject: use4template_dirs[subject] / f"{subject}_asym-brain.nii.gz"
    for subject in used_subj_names
}
asym_mask_paths = {
    subject: use4template_dirs[subject] / f"{subject}_asym-mask.nii.gz"
    for subject in used_subj_names
}

# %%
# Load and process the final average template

template_img = load_nii(final_template_path, as_array=True, as_numpy=True)
template_mask = load_nii(final_mask_path, as_array=True, as_numpy=True)
target_size = max(template_img.shape)
template_img = linear_rescale(
    template_img,
    template_img,
    source_mask=template_mask,
    target_mask=template_mask,
)

template_gmag = gradient_magnitude(template_img, sigma=1)

template_img, pad_sizes = pad_with_zeros(template_img, target=target_size)
template_mask, _ = pad_with_zeros(template_mask, target=target_size)
template_gmag, _ = pad_with_zeros(template_gmag, target=target_size)

print(f"Loaded and processed final template image from {final_template_path}")

# Save the padded images for later use
vox_sizes = [config["resolution_um"] * 1e-3] * 3  # convert to mm
padded_template_path = plots_dir / "final_template_padded.nii.gz"
save_nii(template_img, vox_sizes, padded_template_path)
print(f"Saved padded image for final template to {padded_template_path}")

# %%
# Load and process the example subject image

example_subject = config["example_subjects"][0]
subject_path = asym_inputs_paths[example_subject]
subject_img = load_nii(subject_path, as_array=True, as_numpy=True)
subject_mask = load_nii(
    asym_mask_paths[example_subject], as_array=True, as_numpy=True
)

subject_img = linear_rescale(
    subject_img,
    template_img,
    source_mask=subject_mask,
    target_mask=template_mask,
)
subject_gmag = gradient_magnitude(subject_img, sigma=1)

subject_img, _ = pad_with_zeros(subject_img, target=target_size)
subject_mask, _ = pad_with_zeros(subject_mask, target=target_size)
subject_gmag, _ = pad_with_zeros(subject_gmag, target=target_size)

# Save the padded image for the example subject
padded_subject_path = plots_dir / f"{example_subject}_padded.nii.gz"
save_nii(subject_img, vox_sizes, padded_subject_path)
print(
    f"Saved padded image for {example_subject} " f"to {padded_subject_path}."
)

# %%
# Collect coronal slices for each asymmetric input image
asym_inputs_slices = collect_coronal_slices(
    asym_inputs_paths, config["show_slices"][0]
)
asym_mask_paths = collect_coronal_slices(
    asym_mask_paths, config["show_slices"][0]
)
asym_inputs_slices = {  # Rescale intensities to final template's range
    label: linear_rescale(
        img,
        template_img,
        source_mask=asym_mask_paths[label],
        target_mask=template_mask,
    )
    for label, img in asym_inputs_slices.items()
}
asym_inputs_gmag_slices = {  # Compute gradient magnitude
    label: gradient_magnitude(img, sigma=1)
    for label, img in asym_inputs_slices.items()
}
print(
    f"Collected coronal slice {config['show_slices'][0]} "
    "from each subject's asymmetric brain image"
)

# %%
# Plot coronal slices for each asymmetric input image

coronal_slice_dict = {
    "_": asym_inputs_slices,
    "_gmag_": asym_inputs_gmag_slices,
}
for label, coronal_slices in coronal_slice_dict.items():
    plot_slices_single_row(
        coronal_slices,
        save_path=plots_dir / f"inputs_asym_brain{label}single_row",
    )
    plot_slices_single_column(
        coronal_slices,
        save_path=plots_dir / f"inputs_asym_brain{label}single_column",
    )

print(
    "Plotted coronal slices for individual subjects' "
    "asymmetric brain images and their gradient magnitudes."
)


# %%
# Plot orthographic views
ortho_params = {
    "show_slices": config["show_slices"],
    "pad_sizes": pad_sizes,
    "mip_attenuation": config["mip_attenuation"],
    "scale_bar": True,
    "resolution": config["resolution_um"] * 1e-3,  # convert to mm
}


image_dict = {
    "final_template": template_img,
    f"{example_subject}": subject_img,
}
gmag_dict = {
    "final_template": template_gmag,
    f"{example_subject}": subject_gmag,
}

for label, img in image_dict.items():
    fig, axs = plot_orthographic(
        img,
        **ortho_params,
        vmin=0,
        vmax=np.percentile(template_img, 99.9),
        save_path=plots_dir / f"{label}_orthographic",
    )
    print(f"Plotted {label} in orthographic view")

    fig, axs = plot_orthographic(
        gmag_dict[label],
        **ortho_params,
        vmin=0,
        vmax=np.percentile(template_gmag, 99.9),
        save_path=plots_dir / f"{label}_gmag_orthographic",
    )
    print(f"Plotted {label}'s gradient magnitude in orthographic view")


# %%
# Plot inset comparison between the final template and example subject
inset_params = config["insets"]
for inset_name, inset_param in inset_params.items():
    plot_file_name = f"{example_subject}_vs_template_inset-{inset_name}"
    plot_inset_comparison(
        img1=(example_subject, subject_img),
        img2=("template", template_img),
        **inset_param,
        save_path=plots_dir / plot_file_name,
        scale_bar=True,
        resolution=config["resolution_um"] * 1e-3,  # convert to mm
    )
print(f"Plotted inset comparison between {example_subject} and template")

# %%

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    assert config_path.exists(), f"Config file not found: {config_path}"
    return yaml.safe_load(open(config_path, "r"))


def setup_directories(config: dict) -> tuple[Path, Path, Path]:
    """Extract and create directories for atlas, template, and plots."""
    # Extract atlas directory
    atlas_forge_dir = Path(config["atlas_forge_dir"])
    species = config["species"]
    atlas_dir = atlas_forge_dir / species
    assert atlas_dir.exists(), f"Atlas directory not found: {atlas_dir}"
    print("atlas directory: ", atlas_dir)

    # Extract template name and directory
    template_name = config["template_name"]
    template_dir = atlas_dir / "templates" / template_name
    assert (
        template_dir.exists()
    ), f"Template directory not found: {template_dir}"
    print("template name: ", template_name)

    # Create output directory for plots
    plots_dir = template_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"Will write outputs to plots directory: {plots_dir}")

    return atlas_dir, template_dir, plots_dir


def collect_template_paths(
    template_dir: Path,
    transform_types: list[str] = ["rigid", "similarity", "affine", "nlin"],
    n_iter: int = 4,
    template_file_name: str = "template_sharpen_shapeupdate.nii.gz",
) -> dict[str, Path]:
    """Collect paths to template files per ßtransform type and iteration."""
    paths = {}
    for transform_type in transform_types:
        transform_type_dir = template_dir / transform_type
        for i in range(n_iter):
            path = transform_type_dir / str(i) / "average" / template_file_name
            assert path.exists(), f"File not found: {path}"
            paths[f"{transform_type} iter-{i}"] = path
    return paths


def collect_coronal_slices(
    file_paths: dict[str, Path], show_coronal_slice: int
) -> dict[str, np.ndarray]:
    """Load and extract coronal slices for all file paths.

    Assumes that each path leads to a NIfTI file, with the first dimension
    corresponding to the coronal plane.
    """
    slices = {}
    for img_name, img_path in file_paths.items():
        img = load_nii(img_path, as_array=False)
        slc = img.slicer[show_coronal_slice : show_coronal_slice + 1, :, :]
        slices[img_name] = slc.get_fdata().squeeze()
    return slices


def compute_vmin_vmax_across_slices(
    slices: dict[str, np.ndarray], vmin_perc: float = 1, vmax_perc: float = 99
) -> tuple[float, float]:
    """Calculate consistent vmin and vmax for all slices."""
    vmin = np.min([np.percentile(slc, vmin_perc) for slc in slices.values()])
    vmax = np.max([np.percentile(slc, vmax_perc) for slc in slices.values()])
    return vmin, vmax


def save_figure(fig: plt.Figure, plots_dir: Path, filename: str):
    """Save figure in both PNG and PDF formats."""
    fig.savefig(plots_dir / f"{filename}.png")
    fig.savefig(plots_dir / f"{filename}.pdf")


def collect_use4template_dirs(
    df: pd.DataFrame,
    atlas_dir: Path,
    resolution: int = 25,
    suffix: str = "orig-asr_N4_aligned_padded_use4template",
) -> dict[str, Path]:
    """Collect paths to directories containing images use for template
    building for each subject."""
    deriv_dir = atlas_dir / "derivatives"
    res_str = f"res-{resolution}um"

    use4template_dirs = {}
    for i, row in df.iterrows():
        subject_id = row["subject_id"]
        subject = f"sub-{subject_id}"
        color = row["color"]
        channel = f"channel-{color}"
        image_id = f"{subject}_{res_str}_{channel}"
        use4template_dir = deriv_dir / subject / f"{image_id}_{suffix}"
        assert use4template_dir.is_dir()
        use4template_dirs[subject] = use4template_dir
    return use4template_dirs


def plot_slices_single_row(
    slices: dict[str, np.ndarray],
    vmin_perc: float = 1,
    vmax_perc: float = 99,
    save_path: Path | None = None,
):
    n_slices = len(slices)
    fig, ax = plt.subplots(1, n_slices, figsize=(2 * n_slices, 2))

    for i, label in enumerate(slices):
        frame = slices[label]
        ax[i].imshow(
            frame,
            cmap="gray",
            vmin=np.percentile(frame, vmin_perc),
            vmax=np.percentile(frame, vmax_perc),
        )
        ax[i].axis("off")
        ax[i].set_title(label, fontsize="x-large")

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0, hspace=0
    )
    if save_path:
        save_dir, save_name = save_path.parent, save_path.name.split(".")[0]
        save_figure(fig, save_dir, save_name)


def plot_slices_single_column(
    slices: dict[str, np.ndarray],
    vmin_perc: float = 1,
    vmax_perc: float = 99,
    save_path: Path | None = None,
):
    n_slices = len(slices)
    fig, ax = plt.subplots(n_slices, 1, figsize=(2, 2 * n_slices))

    for i, label in enumerate(slices):
        frame = slices[label]
        ax[i].imshow(
            frame,
            cmap="gray",
            vmin=np.percentile(frame, vmin_perc),
            vmax=np.percentile(frame, vmax_perc),
        )
        ax[i].set_ylabel(label, fontsize="x-large")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        for spine in ax[i].spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.99, bottom=0.01, wspace=0, hspace=0
    )
    if save_path:
        save_dir, save_name = save_path.parent, save_path.name.split(".")[0]
        save_figure(fig, save_dir, save_name)


def pad_with_zeros(
    stack: np.ndarray, target: int = 512
) -> tuple[np.ndarray, list[int]]:
    """Pad the stack with zeros to reach the target size in all dimensions."""
    pad_sizes = [(target - s) // 2 for s in stack.shape]
    padded_stack = np.pad(
        stack,
        (
            (pad_sizes[0], pad_sizes[0]),
            (pad_sizes[1], pad_sizes[1]),
            (pad_sizes[2], pad_sizes[2]),
        ),
        mode="constant",
    )
    return padded_stack, pad_sizes

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image import load_nii
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def grey_cmap_transparent(n: int = 10) -> ListedColormap:
    """Create a custom colormap based on 'gray' but with the bottom
    ``n`` values set to be transparent."""
    cmap = plt.get_cmap("gray").copy()
    colors = cmap(np.linspace(0, 1, cmap.N))
    # Set the alpha channel of the bottom values to 0 (transparent)
    colors[:n, -1] = 0
    return ListedColormap(colors)


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
    file_paths: dict[str, Path], slice_idx: int
) -> dict[str, np.ndarray]:
    """Load and extract coronal slices for all file paths.

    Assumes that each path leads to a NIfTI file, with the first dimension
    corresponding to the coronal plane.
    """
    slices = {}
    for img_name, img_path in file_paths.items():
        img = load_nii(img_path, as_array=False)
        slc = img.slicer[slice_idx : slice_idx + 1, :, :]
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
    n_transparent: int = 0,
    save_path: Path | None = None,
):
    """Plot slices in a single row."""
    n_slices = len(slices)
    fig, ax = plt.subplots(1, n_slices, figsize=(2 * n_slices, 2))

    for i, label in enumerate(slices):
        frame = slices[label]
        ax[i].imshow(
            frame,
            cmap=grey_cmap_transparent(n_transparent),
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
    n_transparent: int = 0,
    save_path: Path | None = None,
):
    """Plot slices in a single column."""
    n_slices = len(slices)
    fig, ax = plt.subplots(n_slices, 1, figsize=(2.4, 1.3 * n_slices))

    for i, label in enumerate(slices):
        frame = slices[label]

        ax[i].imshow(
            frame,
            cmap=grey_cmap_transparent(n_transparent),
            vmin=np.percentile(frame, vmin_perc),
            vmax=np.percentile(frame, vmax_perc),
        )
        ax[i].set_ylabel(label, fontsize="x-large")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        for spine in ax[i].spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.15, right=0.95, top=0.99, bottom=0.01, wspace=0, hspace=0
    )
    if save_path:
        save_dir, save_name = save_path.parent, save_path.name.split(".")[0]
        save_figure(fig, save_dir, save_name)


def pad_with_zeros(
    stack: np.ndarray, target: int = 512
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Pad the stack with zeros to reach the target size in all dimensions."""
    pad_sizes = []
    for s in stack.shape:
        delta = target - s
        if delta < 0:
            raise ValueError(
                f"Cannot pad dimension of size {s} to a "
                f"smaller target size {target}."
            )
        left_pad = delta // 2
        right_pad = delta - left_pad
        pad_sizes.append((left_pad, right_pad))

    padded_stack = np.pad(
        stack,
        pad_sizes,
        mode="constant",
    )
    return padded_stack, pad_sizes


def plot_orthographic(
    img: np.ndarray,
    show_slices: list[int] | None = None,
    pad_sizes: list[tuple[int, int]] | None = None,
    mip_attenuation: float = 0.01,
    save_path: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot orthographic views of a 3D image,
    including a maximum intensity projection (MIP)."""

    sc = AnatomicalSpace("ASR", shape=img.shape)
    if pad_sizes is None:
        pad_sizes = [(0, 0)] * img.ndim

    # Default to the middle slice along each axis
    if show_slices is None:
        show_slices = [img.shape[i] // 2 for i in range(img.ndim)]

    fig, axs = plt.subplots(1, 4, figsize=(14, 4))

    v_axis = sc.get_axis_idx("vertical")
    mip, _ = _compute_attenuated_mip(img, v_axis, mip_attenuation)
    sections = [s.capitalize() for s in sc.sections] + ["Top view"]
    axis_labels = [*sc.axis_labels, sc.axis_labels[v_axis]]
    slice_idxs = [s + p[0] for s, p in zip(show_slices, pad_sizes)]
    frames = [img.take(s, axis=i) for i, s in enumerate(slice_idxs)]
    frames.append(mip)

    for j, (view, labels) in enumerate(zip(sections, axis_labels)):
        ax = axs[j]
        ax.imshow(
            frames[j],
            cmap="gray",
            aspect="equal",
            origin="upper",
            vmin=np.percentile(img, 1),
            vmax=np.percentile(img, 99.9),
        )
        ax.set_title(view)
        ax.set_ylabel(labels[0])
        ax.set_xlabel(labels[1])

        # Indicate location of orthogonal slices with lines
        if j < 3:
            h, v = sc.index_pairs[j]
            ax.axhline(slice_idxs[h], color="r", linestyle="--", alpha=0.5)
            ax.axvline(slice_idxs[v], color="r", linestyle="--", alpha=0.5)

        ax = _clear_spines_and_ticks(ax)

    fig.subplots_adjust(
        left=0.025, right=0.975, top=0.95, bottom=0.05, wspace=0.1, hspace=0
    )
    if save_path:
        save_figure(fig, save_path.parent, save_path.name.split(".")[0])
    return fig, axs


def plot_inset_comparison(
    img1: tuple[str, np.ndarray],
    img2: tuple[str, np.ndarray],
    z: int,
    y_min: int,
    x_min: int,
    size: int,
    save_path: Path | None = None,
):
    """Plot the same inset from two images side by side.

    A 2D square inset is extracted from the specified z-slice of each image
    and displayed side by side. The inset extents are
    ``(y_min, y_min + size, x_min, x_min + size)``. The axis
    order is assumed to be (z, y, x) in ASR orientation.

    Parameters
    ----------
    img1 : tuple[str, np.ndarray]
        Tuple containing the image name and 3D image data for the first image.
    img2 : tuple[str, np.ndarray]
        Tuple containing the image name and 3D image data for the second image.
    z : int
        Index of the z-slice to extract the inset from.
    y_min : int
        Minimum y-coordinate of the inset.
    x_min : int
        Minimum x-coordinate of the inset.
    size : int
        Size of the square inset to extract in pixels.
        (same in y and x dimensions).
    save_path : Path, optional
        Path to save the figure, by default None.

    Returns
    -------
    tuple
        Figure and axes objects.
    """
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    for i, img_tuple in enumerate([img1, img2]):
        name, img = img_tuple
        img_inset = _extract_inset(img, z, y_min, x_min, size)
        ax = axs[i]

        vmin, vmax = np.percentile(img, (1, 99.9))
        ax.imshow(
            img_inset,
            cmap="gray",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axis("off")
        ax.set_title(name)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path.parent, save_path.name.split(".")[0])
    return fig, axs


def _compute_attenuated_mip(
    img: np.ndarray, axis: int, attenuation_factor: float
) -> tuple[np.ndarray, str]:
    """Compute the maximum intensity projection (MIP) with attenuation.

    If the image is zero-padded, attenuation is only applied within the
    non-zero region along the specified axis.

    Parameters
    ----------
    img : np.ndarray
        Image volume.
    axis : int
        Axis along which to compute the MIP.
    attenuation_factor : float
        Attenuation factor for the MIP. 0 means no attenuation.

    Returns
    -------
    tuple[np.ndarray, str]
        MIP image and label. The label is "MIP" if no attenuation is applied,
        and "MIP (attenuated)" otherwise.
    """

    mip_label = "MIP"

    if attenuation_factor < 0:
        raise ValueError("Attenuation factor must be non-negative.")

    if attenuation_factor < 1e-6:
        # If the factor is too small, skip attenuation
        mip = np.max(img, axis=axis)
        return mip, mip_label

    # Find the non-zero bounding box along the specified axis
    other_axes = tuple(i for i in range(img.ndim) if i != axis)
    non_zero_mask = np.any(img != 0, axis=other_axes)
    non_zero_indices = np.nonzero(non_zero_mask)[0]
    start, end = non_zero_indices[0], non_zero_indices[-1] + 1

    # Trim the image along the attenuation axis (get rid of zero-padding)
    slices = [slice(None)] * img.ndim
    slices[axis] = slice(start, end)
    trimmed_img = img[tuple(slices)]

    # Apply attenuation to the trimmed image
    attenuation = np.exp(
        -attenuation_factor * np.arange(trimmed_img.shape[axis])
    )
    attenuation_shape = [1] * trimmed_img.ndim
    attenuation_shape[axis] = trimmed_img.shape[axis]
    attenuation = attenuation.reshape(attenuation_shape)
    attenuated_img = trimmed_img.astype(np.float32) * attenuation

    # Compute and return the attenuated MIP
    mip = np.max(attenuated_img, axis=axis)
    mip_label += " (attenuated)"

    return mip, mip_label


def _clear_spines_and_ticks(ax: plt.Axes) -> plt.Axes:
    """Clear spines and ticks from a matplotlib axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def _extract_inset(
    img: np.ndarray, z: int, y_min: int, x_min: int, size: int
) -> np.ndarray:
    """Extract a 2D square inset from a 3D image.

    Axis order is assumed to be (z, y, x) in ASR orientation and
    indices are expected to be given in pixel units.

    Parameters
    ----------
    img : np.ndarray
        3D image from which to extract the inset.
    z : int
        Index of the z-slice to extract the inset from.
    y_min : int
        Minimum y-coordinate of the inset.
    x_min : int
        Minimum x-coordinate of the inset.
    size : int
        Size of the square inset to extract in pixels.
        (same in y and x dimensions)."""
    return img[z, y_min : y_min + size, x_min : x_min + size]

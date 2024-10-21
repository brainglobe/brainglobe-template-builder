from pathlib import Path
from typing import Literal

import numpy as np
from brainglobe_space import AnatomicalSpace
from matplotlib import pyplot as plt


def plot_orthographic(
    img: np.ndarray,
    anat_space: str = "ASR",
    show_slices: tuple[int, int, int] | None = None,
    mip_attenuation: float = 0.01,
    save_path: Path | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot image volume in three orthogonal views, plus a surface rendering.

    The function assumes isotropic voxels (otherwise the proportions of the
    image will be distorted). The surface rendering is a maximum intensity
    projection (MIP) along the vertical (superior-inferior) axis.

    Parameters
    ----------
    img : np.ndarray
        Image volume to plot.
    anat_space : str, optional
        Anatomical space of of the image volume according to the Brainglobe
        definition (origin and order of axes), by default "ASR".
    show_slices : tuple, optional
        Which slice to show per dimension. If None (default), show the middle
        slice along each dimension.
    mip_attenuation : float, optional
        Attenuation factor for the MIP, by default 0.01.
        A value of 0 means no attenuation.
    save_path : Path, optional
        Path to save the plot, by default None (no saving).
    **kwargs
        Additional keyword arguments to pass to ``matplotlib.pyplot.imshow``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes objects

    """

    space = AnatomicalSpace(anat_space)
    vertical_axis = space.get_axis_idx("vertical")

    # Get middle slices if not specified
    if show_slices is None:
        slices_list = [s // 2 for s in img.shape]
    else:
        slices_list = list(show_slices)

    # Pad the image with zeros to make it cubic
    img, pad_sizes = _pad_with_zeros(img, target=max(img.shape))
    slices_list = [s + pad_sizes[i] for i, s in enumerate(slices_list)]

    # Compute (attenuated) MIP along the vertical axis
    mip, mip_label = _compute_attenuated_mip(
        img, vertical_axis, mip_attenuation
    )

    # Create figure with 4 subplots (3 orthogonal views + MIP)
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    views = [img.take(slc, axis=i) for i, slc in enumerate(slices_list)]
    views.append(mip)
    axis_labels = [*space.axis_labels, space.axis_labels[vertical_axis]]
    section_names = [s.capitalize() for s in space.sections] + [mip_label]

    kwargs = _set_imshow_defaults(img, kwargs)

    for j, (section, labels) in enumerate(zip(section_names, axis_labels)):
        ax = axs[j]
        ax.imshow(views[j], **kwargs)
        ax.set_title(section)
        ax.set_ylabel(labels[0])
        ax.set_xlabel(labels[1])
        ax = _clear_spines_and_ticks(ax)
    plt.tight_layout()

    if save_path:
        _save_and_close_figure(
            fig, save_path.parent, save_path.name.split(".")[0]
        )
    return fig, axs


def plot_grid(
    img: np.ndarray,
    anat_space="ASR",
    section: Literal["frontal", "horizontal", "sagittal"] = "frontal",
    n_slices: int = 12,
    save_path: Path | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot image volume as a grid of slices along a given anatomical section.

    Parameters
    ----------
    img : np.ndarray
        Image volume to plot.
    anat_space : str, optional
        Anatomical space of of the image volume according to the Brainglobe
        definition (origin and order of axes), by default "ASR".
    section : str, optional
        Section to show, must be one of "frontal", "horizontal", or "sagittal",
        by default "frontal".
    n_slices : int, optional
        Number of slices to show, by default 12. Slices will be evenly spaced,
        starting from the first and ending with the last slice.
    save_path : Path, optional
        Path to save the plot, by default None (no saving).
    **kwargs
        Additional keyword arguments to pass to ``matplotlib.pyplot.imshow``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes objects

    """
    space = AnatomicalSpace(anat_space)
    section_to_axis = {  # Mapping of section names to space axes
        "frontal": "sagittal",
        "horizontal": "vertical",
        "sagittal": "frontal",
    }
    axis_idx = space.get_axis_idx(section_to_axis[section])

    # Ensure n_slices is not greater than the number of slices in the image
    n_slices = min(n_slices, img.shape[axis_idx])
    # ensure first and last slices are included
    show_slices = np.linspace(0, img.shape[axis_idx] - 1, n_slices, dtype=int)

    # Get slices along the specified axis and arrange them in a grid
    grid_img = _grid_from_slices(
        [img.take(slc, axis=axis_idx) for slc in show_slices]
    )

    # Plot the grid image
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    kwargs = _set_imshow_defaults(img, kwargs)
    ax.imshow(grid_img, **kwargs)

    section_name = section.capitalize()
    ax.set_title(f"{section_name} slices")
    ax.set_xlabel(space.axis_labels[axis_idx][1])
    ax.set_ylabel(space.axis_labels[axis_idx][0])
    ax = _clear_spines_and_ticks(ax)
    plt.tight_layout()

    if save_path:
        _save_and_close_figure(
            fig, save_path.parent, save_path.name.split(".")[0]
        )
    return fig, ax


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


def _save_and_close_figure(fig: plt.Figure, plots_dir: Path, filename: str):
    """Save figure in both PNG and PDF formats and close it."""
    fig.savefig(plots_dir / f"{filename}.png")
    fig.savefig(plots_dir / f"{filename}.pdf")
    plt.close(fig)


def _clear_spines_and_ticks(ax: plt.Axes) -> plt.Axes:
    """Clear spines and ticks from a matplotlib axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def _set_imshow_defaults(img: np.ndarray, kwargs: dict) -> dict:
    """Set default values for imshow keyword arguments.

    These apply only if the user does not provide them explicitly.
    """
    if "vmin" not in kwargs and "vmax" not in kwargs:
        vmin, vmax = _auto_adjust_contrast(img)
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

    kwargs.setdefault("cmap", "gray")
    kwargs.setdefault("aspect", "equal")
    return kwargs


def _grid_from_slices(slices: list[np.ndarray]) -> np.ndarray:
    """Create a grid image from a list of 2D slices.

    The number of rows is automatically determined based on the square root
    of the number of slices, rounded up.

    Parameters
    ----------
    slices : list[np.ndarray]
        List of 2D slices to concatenate.

    Returns
    -------
    np.ndarray
        A 2D image, with the input slices arranged in a grid.

    """

    n_slices = len(slices)
    slice_height, slice_width = slices[0].shape

    # Form image mosaic grid by concatenating slices
    n_rows = int(np.ceil(np.sqrt(n_slices)))
    n_cols = int(np.ceil(n_slices / n_rows))
    grid_img = np.zeros(
        (n_rows * slice_height, n_cols * slice_width),
    )
    for i, slice in enumerate(slices):
        row = i // n_cols
        col = i % n_cols
        grid_img[
            row * slice_height : (row + 1) * slice_height,
            col * slice_width : (col + 1) * slice_width,
        ] = slice

    return grid_img


def _pad_with_zeros(
    img: np.ndarray, target: int = 512
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Pad the volume with zeros to reach the target size in all dimensions."""
    pad_sizes = [(target - s) // 2 for s in img.shape]
    padded_img = np.pad(
        img,
        (
            (pad_sizes[0], pad_sizes[0]),
            (pad_sizes[1], pad_sizes[1]),
            (pad_sizes[2], pad_sizes[2]),
        ),
        mode="constant",
    )
    return padded_img, tuple(pad_sizes)


def _auto_adjust_contrast(img, lower_percentile=1, upper_percentile=99):
    """Adjust contrast of an image using percentile-based scaling."""
    # Mask near-zero voxels to exclude background
    if np.issubdtype(img.dtype, np.integer):
        background_threshold = 1
    else:
        background_threshold = np.finfo(img.dtype).eps

    brain_mask = img > background_threshold

    # Exclude bright artifacts
    vmax = np.percentile(img[brain_mask], upper_percentile)
    artifact_mask = img <= vmax
    combined_mask = brain_mask & artifact_mask

    # Compute vmin and vmax
    vmin = np.percentile(img[combined_mask], lower_percentile)
    vmax = np.percentile(img[combined_mask], upper_percentile)

    return vmin, vmax

from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image.load import load_any, read_with_dask
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.plots import plot_orthographic
from brainglobe_template_builder.preproc.transform_utils import (
    downsample_anisotropic_stack_to_isotropic,
)
from brainglobe_template_builder.validate import validate_input_csv


def _get_subject_path(
    standardised_dir: Path,
    subject_id: str,
    output_vox_size: float,
    mask: bool = False,
) -> Path:
    """Get path to standardised nifti file for subject_id, and create any
    required parent directories.

    For example, for a subject with ID 002 at voxel size 2.1 micrometre,
    return:
    "{standardised_dir}/sub-002/sub-002_res-2x2x2um{_mask}_origin-asr.nii.gz".
    """

    # round output vox sizes to nearest int (we don't want decimal
    # points in the filename)
    rounded_size = round(output_vox_size)
    resolution_string = f"res-{rounded_size}x{rounded_size}x{rounded_size}um"

    subject_dir = standardised_dir / f"sub-{subject_id}"
    file_name = f"sub-{subject_id}_{resolution_string}"
    if mask:
        dest_path = subject_dir / f"{file_name}_mask_origin-asr.nii.gz"
    else:
        dest_path = subject_dir / f"{file_name}_origin-asr.nii.gz"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    return dest_path


def _process_image(
    image_path: Path,
    output_path: Path,
    origin: str,
    input_vox_sizes: list[float],
    output_vox_size: float,
    mask: bool = False,
) -> np.ndarray:
    """
    Standardise a single source image.

    Converts to ASR orientation, and downsamples to an isotropic
    resolution of [output_vox_size, output_vox_size, output_vox_size].
    The processed image is written to 'output_path' and returned as
    a numpy array.

    Parameters
    ----------
    image_path : Path
        Path to image to standardise.
    output_path : Path
        Path to write processed image to.
    origin : str
        3-letter anatomical orientation code (e.g., PSL, LSP, RAS)
    input_vox_sizes : list[float]
        Input voxel sizes in microns - matching anatomical axis order.
    output_vox_size : float
        Output voxel size in microns.
    mask : bool, optional
        Whether the input image is a mask, by default False
    """

    if (np.array(input_vox_sizes) == output_vox_size).all():
        image = load_any(image_path)
    else:
        # downsample to target resolution via dask to handle
        # out-of-memory images
        image_dask = read_with_dask(image_path)
        image = downsample_anisotropic_stack_to_isotropic(
            image_dask, input_vox_sizes, output_vox_size, mask=mask
        )

    # re-orient to ASR
    space = AnatomicalSpace(origin)
    image = space.map_stack_to("asr", image)

    output_vox_sizes = [output_vox_size, output_vox_size, output_vox_size]
    vox_sizes_mm = [vox_size * 0.001 for vox_size in output_vox_sizes]
    save_as_asr_nii(image, vox_sizes=vox_sizes_mm, dest_path=output_path)

    return image


def _write_QC_plots(
    standardised_qc_dir: Path,
    subject_id: str,
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    """Write QC plots for the standardised image and (optional) mask.

    Parameters
    ----------
    standardised_qc_dir : Path
        Path to standardised QC directory.
    subject_id : str
        Subject id.
    image : np.ndarray
        Standardised image (output from _process_image).
    mask : np.ndarray | None
        Optional standardised mask (output from _process_image)
    """

    # Orthographic image plot
    plot_path = standardised_qc_dir / f"sub-{subject_id}-QC-orthographic.png"
    plot_orthographic(image, anat_space="ASR", save_path=plot_path)

    # Orthographic image + mask plot
    if mask is not None:
        plot_path = (
            standardised_qc_dir / f"sub-{subject_id}-mask-QC-orthographic.png"
        )
        plot_orthographic(
            image,
            overlay=mask,
            anat_space="ASR",
            overlay_is_mask=True,
            save_path=plot_path,
        )


def _process_subject(
    subject_row: pd.Series,
    standardised_dir: Path,
    standardised_qc_dir: Path,
    output_vox_size: float | None = None,
) -> tuple[Path | None, ...]:
    """Standardise source images of an individual subject.

    A directory is created inside 'standardised_dir' for the subject
    containing:
    - the standardised nifti image file
    - the standardised nifti mask file (if a mask_filepath was provided)

    QC plots are written for the standardised images/masks into
    'standardised_qc_dir' to verify the ASR orientation.

    Parameters
    ----------
    subject_row : pd.Series
        Subject row from input csv file (in standardised format
        defined in the atlas-forge documentation).
    standardised_dir : Path
        Directory to write standardised subject images/masks to.
    standardised_qc_dir : Path
        Directory to write subject QC plots to.
    output_vox_size : float | None, optional
        Output voxel size in micrometre. Images will be downsampled to
        an isotropic resolution of
        output_vox_size x output_vox_size x output_vox_size. If not
        provided, input images will be left as-is (they must have
        isotropic resolution!).

    Returns
    -------
    tuple[Path | None, ...]
        Returns: (
            Path to standardised nifti image file,
            Path to standardised nifti mask file - if provided, otherwise None
        )
    """

    subject_id = subject_row.subject_id
    input_vox_sizes = [
        subject_row.resolution_0,
        subject_row.resolution_1,
        subject_row.resolution_2,
    ]

    # Enforce input images must have isotropic voxel size, if no
    # downsampling
    if output_vox_size is None:
        if len(set(input_vox_sizes)) != 1:
            raise ValueError(
                f"Subject id: {subject_id} has anisotropic voxel size: "
                f"{input_vox_sizes}. Pass an output_vox_size to re-sample it."
            )
        output_vox_size = input_vox_sizes[0]

    # Get path of image + (optional) mask
    paths_to_process: dict[str, Path | None] = {}
    paths_to_process["image"] = Path(subject_row.filepath)

    if ("mask_filepath" in subject_row) and pd.notna(
        subject_row.mask_filepath
    ):
        paths_to_process["mask"] = Path(subject_row.mask_filepath)
    else:
        paths_to_process["mask"] = None

    # Process image + (optional) mask
    output_paths: dict[str, Path | None] = {}
    output_images: dict[str, np.ndarray | None] = {}
    for image_name, image_path in paths_to_process.items():

        if image_path is None:
            output_paths[image_name] = None
            output_images[image_name] = None
            continue

        if image_name == "mask":
            is_mask = True
        else:
            is_mask = False

        output_path = _get_subject_path(
            standardised_dir, subject_id, output_vox_size, mask=is_mask
        )
        processed_image = _process_image(
            image_path,
            output_path,
            subject_row.origin,
            input_vox_sizes,
            output_vox_size,
            mask=is_mask,
        )

        output_paths[image_name] = output_path
        output_images[image_name] = processed_image

    # Write QC plots for image and (optionally) mask overlaid on image
    _write_QC_plots(
        standardised_qc_dir,
        subject_id,
        image=output_images["image"],
        mask=output_images["mask"],
    )

    return (output_paths["image"], output_paths["mask"])


def standardise(
    source_csv: Path, output_dir: Path, output_vox_size: float | None = None
) -> None:
    """Standardise source images to the same resolution and orientation (ASR).

    A 'standardised' directory is created inside 'output_dir', containing a
    folder for each subject id with:
    - the standardised nifti image file
    - the standardised nifti mask file (if a mask_filepath was provided)

    A 'standardised-QC' directory is created inside 'output_dir' containing:
    - a QC plot for every subject image + mask to verify ASR orientation

    At the top level of the 'standardised' dir, a csv file is created
    summarising the properties / locations of the standardised image files.

    Parameters
    ----------
    source_csv : Path
        Source csv with one row per subject id. Should be in the standard
        format defined in the atlas-forge documentation.
    output_dir : Path
        Output directory to write to.
    output_vox_size : float | None, optional
        Output voxel size in micrometre. Images will be downsampled to
        an isotropic resolution of
        output_vox_size x output_vox_size x output_vox_size. If not
        provided, input images will be left as-is (they must have
        isotropic resolution!).
    """

    validate_input_csv(source_csv)
    source_df = pd.read_csv(source_csv)

    standardised_dir = output_dir / "standardised"
    standardised_qc_dir = output_dir / "standardised-QC"
    for directory in [standardised_dir, standardised_qc_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if "use" in source_df:
        source_df = source_df[source_df.use]

    processed_image_paths = []
    processed_mask_paths = []
    for _, row in source_df.iterrows():
        image_path, mask_path = _process_subject(
            row, standardised_dir, standardised_qc_dir, output_vox_size
        )
        processed_image_paths.append(image_path)
        processed_mask_paths.append(mask_path)

    # Make output csv for processed images
    output_df = source_df.copy()
    output_df.origin = "ASR"
    output_df.filepath = processed_image_paths

    if "mask_filepath" in output_df:
        output_df.mask_filepath = processed_mask_paths

    if output_vox_size is not None:
        output_df.resolution_0 = output_vox_size
        output_df.resolution_1 = output_vox_size
        output_df.resolution_2 = output_vox_size

    output_df.to_csv(standardised_dir / "standardised_images.csv", index=False)

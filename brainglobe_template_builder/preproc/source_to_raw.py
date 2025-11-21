from pathlib import Path

import pandas as pd
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image.load import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.plots import plot_orthographic
from brainglobe_template_builder.validate import validate_input_csv


def _get_subject_path(
    raw_dir: Path, subject_id: str, output_vox_sizes: list[float]
) -> Path:
    """Get path to standardised nifti file for subject_id, and create any
    required parent directories."""

    # round output vox sizes to nearest int (we don't want decimal
    # points in the filename)
    rounded_sizes = [round(vox_size) for vox_size in output_vox_sizes]
    resolution_string = (
        f"res-{rounded_sizes[0]}x{rounded_sizes[1]}x{rounded_sizes[2]}um"
    )
    dest_path = (
        raw_dir
        / f"sub-{subject_id}"
        / f"sub-{subject_id}_{resolution_string}_origin-asr.nii.gz"
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    return dest_path


def _process_subject(
    subject_row: pd.Series,
    raw_dir: Path,
    output_vox_size: float | None = None,
) -> Path:
    """Standardise source image of an individual subject.

    A directory is created inside 'output_dir' for the subject containing:
    - the standardised nifti image file
    - a QC plot to verify the ASR orientation

    Parameters
    ----------
    subject_row : pd.Series
        Subject row from input csv file (in standardised format
        defined in the atlas-forge documentation).
    raw_dir : Path
        Raw directory to write subject files to.
    output_vox_size : float | None, optional
        Output voxel size in micrometre. Images will be downsampled to
        an isotropic resolution of
        output_vox_size x output_vox_size x output_vox_size. If not
        provided, input images will be left as-is (they must have
        isotropic resolution!).

    Returns
    -------
    Path
        Path to standardised nifti file.
    """

    image = load_any(subject_row.source_filepath)
    subject_id = subject_row.subject_id

    # downsample to target isotropic resolution
    if output_vox_size is None:
        output_vox_sizes = [
            subject_row.resolution_x,
            subject_row.resolution_y,
            subject_row.resolution_z,
        ]
        if len(set(output_vox_sizes)) != 1:
            raise ValueError(
                f"Subject id: {subject_id} has anisotropic voxel size: "
                f"{output_vox_sizes}. Pass an output_vox_size to re-sample it."
            )

    else:
        output_vox_sizes = [output_vox_size, output_vox_size, output_vox_size]
        # TODO - downsample

    # re-orient to ASR
    space = AnatomicalSpace(subject_row.origin)
    image = space.map_stack_to("asr", image)

    # Get path to output image file, incorporating output
    # resolution into filename
    dest_path = _get_subject_path(raw_dir, subject_id, output_vox_sizes)

    # save QC plot of re-oriented image
    plot_path = dest_path.parent / f"sub-{subject_id}-QC-standardised.png"
    plot_orthographic(image, anat_space="ASR", save_path=plot_path)

    vox_sizes_mm = [vox_size * 0.001 for vox_size in output_vox_sizes]
    save_as_asr_nii(image, vox_sizes=vox_sizes_mm, dest_path=dest_path)

    return dest_path


def source_to_raw(
    source_csv: Path, output_dir: Path, output_vox_size: float | None = None
) -> None:
    """Standardise source images to the same resolution and orientation (ASR).

    A 'raw' directory is created inside 'output_dir', containing a folder for
    each subject id with:
    - the standardised nifti image file
    - a QC plot to verify the ASR orientation

    At the top level of the 'raw' dir, a csv file is created summarising the
    properties / locations of the standardised image files.

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

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if "use" in source_df:
        source_df = source_df[source_df.use is True]

    processed_paths = []
    for _, row in source_df.iterrows():
        nifti_path = _process_subject(row, raw_dir, output_vox_size)
        processed_paths.append(nifti_path)

    # Make output csv for processed images
    output_df = source_df.copy()
    output_df.origin = "ASR"
    output_df.source_filepath = processed_paths

    if output_vox_size is not None:
        output_df.resolution_z = output_vox_size
        output_df.resolution_y = output_vox_size
        output_df.resolution_x = output_vox_size

    output_df.to_csv(output_dir / "raw_images.csv", index=False)

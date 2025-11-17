from pathlib import Path

import pandas as pd
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image.load import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.plots import plot_orthographic


def _process_subject(
    subject_row: pd.Series,
    output_dir: Path,
    output_vox_size: float | None = None,
) -> Path:

    image = load_any(subject_row.source_filepath)

    subject_id = subject_row.subject_id
    resolution_string = (
        f"res-{output_vox_size}x{output_vox_size}x{output_vox_size}um"
    )
    dest_path = (
        output_dir
        / f"sub-{subject_id}"
        / f"sub-{subject_id}_{resolution_string}_origin-asr.nii.gz"
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # downsample to target isotropic resolution
    if output_vox_size is not None:
        pass  # TODO - downsample

    # re-orient to ASR
    space = AnatomicalSpace(subject_row.origin)
    image = space.map_stack_to("asr", image)

    # save QC plot of re-oriented image
    plot_path = dest_path.parent / f"sub-{subject_id}-QC-standardised.png"
    plot_orthographic(image, anat_space="ASR", save_path=plot_path)

    vox_sizes_mm = [
        subject_row.resolution_x * 0.001,
        subject_row.resolution_y * 0.001,
        subject_row.resolution_z * 0.001,
    ]
    save_as_asr_nii(image, vox_sizes=vox_sizes_mm, dest_path=dest_path)

    return dest_path


def source_to_raw(
    input_csv: Path, output_dir: Path, output_vox_size: float | None = None
) -> None:

    input_df = pd.read_csv(input_csv)
    # TODO - Validate input csv

    if "use" in input_df:
        input_df = input_df[input_df.use is True]

    processed_paths = []
    for _, row in input_df.iterrows():
        nifti_path = _process_subject(row, output_dir, output_vox_size)
        processed_paths.append(nifti_path)

    # Make output csv for processed images
    output_df = input_df.copy()
    output_df.origin = "ASR"
    output_df.resolution_z = output_vox_size
    output_df.resolution_y = output_vox_size
    output_df.resolution_x = output_vox_size
    output_df.source_filepath = processed_paths
    output_df.to_csv(output_dir / "raw_images.csv", index=False)

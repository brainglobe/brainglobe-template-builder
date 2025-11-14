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

    for _, row in input_df.iterrows():

        if ("use" in row) and (row.use is False):
            continue

        nifti_path = _process_subject(row, output_dir, output_vox_size)
        print(nifti_path)

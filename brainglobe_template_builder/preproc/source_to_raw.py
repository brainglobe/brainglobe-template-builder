from pathlib import Path

import pandas as pd
from brainglobe_utils.IO.image.load import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii


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

    # downsample to target isotropic resolution

    # re-orient

    vox_sizes_mm = [
        subject_row.resolution_x * 0.001,
        subject_row.resolution_y * 0.001,
        subject_row.resolution_z * 0.001,
    ]
    save_as_asr_nii(image, vox_sizes=vox_sizes_mm, dest_path=dest_path)

    return dest_path


def source_to_raw(
    input_csv: Path, output_dir: Path, output_resolution: float | None = None
) -> None:

    input_df = pd.read_csv(input_csv)
    # TODO - Validate input csv

    for _, row in input_df.iterrows():

        if ("use" in row) and (row.use is False):
            continue

        nifti_path = _process_subject(row, output_dir, output_resolution)
        print(nifti_path)

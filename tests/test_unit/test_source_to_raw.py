from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from brainglobe_utils.IO.image.load import load_nii
from brainglobe_utils.IO.image.save import save_any
from numpy.typing import NDArray

from brainglobe_template_builder.preproc.source_to_raw import source_to_raw


@pytest.fixture()
def stack() -> NDArray[np.float64]:
    """Create 50x50x50 stack with a small off-centre object (value 0.5).

    The object is off-centre, so we can identify the effects of
    re-orientation to ASR.
    """
    stack = np.zeros((50, 50, 50))
    stack[5:30, 5:30, 5:30] = 0.5
    return stack


@pytest.fixture()
def mask() -> NDArray[np.float64]:
    """Create 50x50x50 binary mask with 31×31×31 centred foreground."""
    mask = np.zeros((50, 50, 50))
    mask[10:41, 10:41, 10:41] = 1
    return mask


@pytest.fixture()
def source_dir(tmp_path) -> Path:
    """Create a temporary 'source' directory."""
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    return source_dir


def create_test_images(
    path: Path, test_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Save test images/masks and add their paths to the
    returned test data dicts."""

    for data in test_data:
        subject_dir = path / data["subject_id"]
        image_path = subject_dir / f"{data['subject_id']}.tiff"
        image_path.parent.mkdir()
        save_any(data["image"], image_path)
        data["source_filepath"] = image_path

        if data["mask"] is not None:
            mask_path = subject_dir / f"{data['subject_id']}_mask.tiff"
            save_any(data["mask"], mask_path)
            data["mask_filepath"] = mask_path

    return test_data


def create_test_csv(path: Path, test_data: list[dict[str, Any]]) -> Path:
    """Creates "source_data" CSV file and returns it's path."""

    # we don't include the image / mask in the csv file,
    # just metadata about them
    df_data = test_data.copy()
    for data in df_data:
        data.pop("image")
        data.pop("mask")

    input_csv = pd.DataFrame(data=df_data)
    csv_path = path / "source_data.csv"
    input_csv.to_csv(csv_path, index=False)
    return csv_path


def write_test_data(source_dir: Path, test_data: list[dict[str, Any]]) -> Path:
    """Write test data, and return the path to the summary source csv."""
    test_data = create_test_images(source_dir, test_data)
    csv_path = create_test_csv(source_dir, test_data)
    return csv_path


@pytest.fixture()
def source_csv_no_masks(source_dir: Path, stack: NDArray[np.float64]) -> Path:
    """Creates source images and csv in temporary directory -
    no mask images."""

    # Create test data for two subjects with different voxel sizes.
    test_data = [
        {
            "image": stack,
            "mask": None,
            "subject_id": "a",
            "resolution_z": 25,
            "resolution_y": 25,
            "resolution_x": 25,
            "origin": "PSL",
        },
        {
            "image": stack,
            "mask": None,
            "subject_id": "b",
            "resolution_z": 10,
            "resolution_y": 10,
            "resolution_x": 10,
            "origin": "LSA",
        },
    ]

    return write_test_data(source_dir, test_data)


@pytest.fixture()
def source_csv_with_masks(
    source_dir: Path, stack: NDArray[np.float64], mask: NDArray[np.float64]
) -> Path:
    """Creates source images and csv in temporary directory -
    with mask images."""
    test_data = [
        {
            "image": stack,
            "mask": mask,
            "subject_id": "a",
            "resolution_z": 25,
            "resolution_y": 25,
            "resolution_x": 25,
            "origin": "PSL",
        },
        {
            "image": stack,
            "mask": None,
            "subject_id": "b",
            "resolution_z": 10,
            "resolution_y": 10,
            "resolution_x": 10,
            "origin": "LSA",
        },
    ]

    return write_test_data(source_dir, test_data)


@pytest.mark.parametrize(
    "source_csv", ["source_csv_no_masks", "source_csv_with_masks"]
)
def test_source_to_raw(request, source_csv):

    source_csv_path = request.getfixturevalue(source_csv)
    output_dir = source_csv_path.parents[1]
    output_vox_size = 50
    output_vox_sizes_mm = (
        output_vox_size * 0.001,
        output_vox_size * 0.001,
        output_vox_size * 0.001,
    )

    source_to_raw(source_csv_path, output_dir, output_vox_size)

    assert (output_dir / "raw").exists()

    # Check subject directories exist with correct files inside
    source_df = pd.read_csv(source_csv_path)
    raw_image_filepaths = []
    raw_mask_filepaths = []

    for _, row in source_df.iterrows():
        subject_id = row.subject_id
        subject_dir = output_dir / "raw" / f"sub-{subject_id}"
        image_path = (
            subject_dir / f"sub-{subject_id}_res-50x50x50um_origin-asr.nii.gz"
        )
        raw_image_filepaths.append(str(image_path))

        assert subject_dir.exists()
        assert image_path.exists()
        assert (
            subject_dir / f"sub-{subject_id}_res-50x50x50um_origin-asr-QC.png"
        ).exists()

        image = load_nii(image_path, as_array=False)
        assert image.header.get_zooms() == output_vox_sizes_mm
        assert image.shape == (50, 50, 50)

        mask_path = (
            subject_dir
            / f"sub-{subject_id}_res-50x50x50um_mask_origin-asr.nii.gz"
        )
        mask_expected = ("mask_filepath" in row) and pd.notna(
            row.mask_filepath
        )
        if mask_expected:
            assert mask_path.exists()

            mask = load_nii(mask_path, as_array=False)
            assert mask.header.get_zooms() == output_vox_sizes_mm
            assert mask.shape == (50, 50, 50)

            raw_mask_filepaths.append(str(mask_path))
        else:
            assert not mask_path.exists()
            raw_mask_filepaths.append(np.nan)

    # Check output csv exists with correct metadata
    output_csv_path = output_dir / "raw" / "raw_images.csv"
    assert output_csv_path.exists()

    output_csv = pd.read_csv(output_csv_path)
    expected_output_csv = pd.DataFrame(
        data={
            "subject_id": source_df.subject_id,
            "resolution_z": output_vox_size,
            "resolution_y": output_vox_size,
            "resolution_x": output_vox_size,
            "origin": "ASR",
            "source_filepath": raw_image_filepaths,
        }
    )

    if "mask_filepath" in source_df:
        expected_output_csv["mask_filepath"] = raw_mask_filepaths

    assert output_csv.equals(expected_output_csv)

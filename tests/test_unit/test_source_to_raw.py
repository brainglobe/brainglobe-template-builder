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
    """Create 50x50x50 stack with 21x21x21 centred object (value 0.5)."""
    stack = np.zeros((50, 50, 50))
    stack[15:36, 15:36, 15:36] = 0.5
    return stack


@pytest.fixture()
def mask() -> NDArray[np.float64]:
    """Create 50x50x50 binary mask with 31×31×31 centred foreground."""
    mask = np.zeros((50, 50, 50))
    mask[10:41, 10:41, 10:41] = 1
    return mask


@pytest.fixture()
def test_data(stack: NDArray[np.float64]) -> list[dict[str, Any]]:
    """Create test data for two subjects with different voxel sizes."""
    return [
        {
            "image": stack,
            "subject_id": "a",
            "resolution_z": 25,
            "resolution_y": 25,
            "resolution_x": 25,
            "origin": "ASR",
        },
        {
            "image": stack,
            "subject_id": "b",
            "resolution_z": 10,
            "resolution_y": 10,
            "resolution_x": 10,
            "origin": "ASR",
        },
    ]


def create_test_images(
    path: Path, test_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Save test images and add their paths to the returned test data dicts."""
    for data in test_data:
        image_path = path / data["subject_id"] / f"{data['subject_id']}.tiff"
        image_path.parent.mkdir()
        save_any(data["image"], image_path)
        data["source_filepath"] = image_path
    return test_data


def create_test_csv(path: Path, test_data: list[dict[str, Any]]) -> Path:
    """Creates "source_data" CSV file and returns it's path."""

    # we don't include the image in the csv file, just metadata about it
    for data in test_data:
        data.pop("image")

    input_csv = pd.DataFrame(data=test_data)
    csv_path = path / "source_data.csv"
    input_csv.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def source_csv_no_masks(
    tmp_path: Path, test_data: list[dict[str, Any]]
) -> Path:
    """Sets up temp directory with "source" test images and csv."""
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    test_data = create_test_images(source_dir, test_data)
    csv_path = create_test_csv(source_dir, test_data)
    return csv_path


def test_source_to_raw_no_masks(source_csv_no_masks):
    output_dir = source_csv_no_masks.parents[1]
    output_vox_size = 50
    output_vox_sizes_mm = (
        output_vox_size * 0.001,
        output_vox_size * 0.001,
        output_vox_size * 0.001,
    )

    source_to_raw(source_csv_no_masks, output_dir, output_vox_size)

    assert (output_dir / "raw").exists()

    # Check subject directories exist with correct files inside
    image_filepaths = []
    subject_ids = ["a", "b"]
    for subject_id in subject_ids:
        subject_dir = output_dir / "raw" / f"sub-{subject_id}"
        image_path = (
            subject_dir / f"sub-{subject_id}_res-50x50x50um_origin-asr.nii.gz"
        )
        image_filepaths.append(str(image_path))

        assert subject_dir.exists()
        assert image_path.exists()
        assert (
            subject_dir / f"sub-{subject_id}_res-50x50x50um_origin-asr-QC.png"
        ).exists()

        image = load_nii(image_path, as_array=False)
        assert image.header.get_zooms() == output_vox_sizes_mm
        assert image.shape == (50, 50, 50)

    # Check output csv exists with correct metadata
    output_csv_path = output_dir / "raw" / "raw_images.csv"
    assert output_csv_path.exists()

    output_csv = pd.read_csv(output_csv_path)
    expected_output_csv = pd.DataFrame(
        data={
            "subject_id": subject_ids,
            "resolution_z": [output_vox_size, output_vox_size],
            "resolution_y": [output_vox_size, output_vox_size],
            "resolution_x": [output_vox_size, output_vox_size],
            "origin": ["ASR", "ASR"],
            "source_filepath": image_filepaths,
        }
    )
    assert output_csv.equals(expected_output_csv)

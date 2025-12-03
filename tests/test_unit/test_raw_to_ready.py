import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from brainglobe_utils.IO.image import load_any
from brainglobe_utils.IO.image.save import save_as_asr_nii
from numpy.testing import assert_raises
from numpy.typing import NDArray

from brainglobe_template_builder.preproc.preproc_config import PreprocConfig
from brainglobe_template_builder.preproc.raw_to_ready import (
    _create_subject_dir,
    _process_subject,
    _save_niftis,
    raw_to_ready,
)


def make_stack(
    offset: int | None = None,
    mask: bool = False,
) -> NDArray[np.float64]:
    """Create a 50x50x50 zeros stack with foreground."""

    shape = [50, 50, 50]
    obj_size = 20
    mask_extra = 5
    value = 1 if mask else 0.5

    stack = np.zeros(shape, dtype=np.float64)
    foreground_size = obj_size + (mask_extra if mask else 0)

    start = [(s - foreground_size) // 2 for s in shape]
    if offset:
        start = [s - offset for s in start]
    end = [s + foreground_size for s in start]

    stack[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = value

    return stack


@pytest.fixture()
def test_stacks() -> dict[str, NDArray[np.float64]]:
    """Create symmetric and asymmetric test images and masks."""
    return {
        "image": make_stack(offset=5),
        "mask": make_stack(mask=True, offset=5),
    }


@pytest.fixture()
def test_data(
    test_stacks: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    """Create test data for two subjects with different voxel sizes."""
    return [
        {
            "subject_id": "test1",
            "image": test_stacks["image"],
            "voxel_size": [1, 1, 1],
            "origin": "ASR",
        },
        {
            "subject_id": "test2",
            "image": test_stacks["image"],
            "voxel_size": [2, 2, 2],
            "origin": "ASR",
        },
    ]


def create_test_images(
    path: Path, test_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Save test images and add their paths to the returned test data dicts."""
    for data in test_data:
        image_path = path / data["subject_id"] / f"{data['subject_id']}.nii.gz"
        image_path.parent.mkdir()
        save_as_asr_nii(data["image"], data["voxel_size"], image_path)
        data["source_filepath"] = image_path
    return test_data


def create_test_csv(path: Path, test_data: list[dict[str, Any]]) -> Path:
    """Creates "raw_data" CSV file and returns it's path."""
    for data in test_data:
        data["resolution_z"] = data["voxel_size"][0]
        data["resolution_y"] = data["voxel_size"][1]
        data["resolution_x"] = data["voxel_size"][2]
        data.pop("voxel_size")
        data.pop("image")
    input_csv = pd.DataFrame(data=test_data)
    csv_path = path / "raw_data.csv"
    input_csv.to_csv(csv_path, index=False)
    return csv_path


def create_test_yaml(path: Path) -> Path:
    """Creates YAML config file and returns it's path."""
    yaml_dict = {
        "output_dir": str(path.resolve()),
        "mask": {
            "gaussian_sigma": 3,
            "threshold_method": "triangle",
            "closing_size": 5,
            "erode_size": 0,
        },
        "pad_pixels": 5,
    }

    config_path = path / "config.yaml"
    with open(config_path, "w") as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False)

    return config_path


@pytest.fixture()
def create_raw_test_data(
    tmp_path: Path, test_data: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Sets up temp directory with "raw" test images, CSV, and config files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    test_data = create_test_images(raw_dir, test_data)
    csv_path = create_test_csv(raw_dir, test_data)
    config_path = create_test_yaml(tmp_path)
    return csv_path, config_path


@pytest.mark.parametrize(
    ["use", "expected_listdir"],
    [
        pytest.param(["false", "false"], {}, id="false, false"),
        pytest.param(["False", "false"], {}, id="False, false"),
        pytest.param(["false", "true"], {"sub-test2"}, id="false, true"),
        pytest.param(["", "false"], {"sub-test1"}, id="empty (true), false"),
        pytest.param(
            ["F", "T"], {"sub-test1", "sub-test2"}, id="F (true), T (true)"
        ),
        pytest.param(
            ["0", "1"], {"sub-test1", "sub-test2"}, id="0 (true), 1 (true)"
        ),
    ],
)
def test_raw_to_ready_use_input(
    use: list[str],
    expected_listdir: set,
    create_raw_test_data: tuple[Path, Path],
) -> None:
    """Test exclusion/inclusion based on optional 'use' column values."""
    csv_path, config_path = create_raw_test_data
    input_df = pd.read_csv(csv_path)
    input_df["use"] = use
    input_df.to_csv(csv_path, index=False)

    der_dir = create_raw_test_data[0].parents[0].parents[0] / "derivatives"

    always_expect = {
        "all_processed_brain_paths.txt",
        "all_processed_mask_paths.txt",
    }

    raw_to_ready(csv_path, config_path)
    assert set(os.listdir(der_dir)) == always_expect.union(expected_listdir)


def test_raw_to_ready(create_raw_test_data: tuple[Path, Path]) -> None:
    """Test that raw_to_ready creates expected directories and files."""
    csv_path, config_path = create_raw_test_data
    raw_to_ready(csv_path, config_path)

    der_dir = create_raw_test_data[0].parents[0].parents[0] / "derivatives"

    assert der_dir.exists()
    assert set(os.listdir(der_dir)) == {
        "all_processed_brain_paths.txt",
        "all_processed_mask_paths.txt",
        "sub-test1",
        "sub-test2",
    }
    for i in [1, 2]:
        assert set(os.listdir(der_dir / f"sub-test{i}")) == {
            f"sub-test{i}-QC-mask.pdf",
            f"sub-test{i}-QC-mask.png",
            f"test{i}_processed.nii.gz",
            f"test{i}_processed_lrflip.nii.gz",
            f"test{i}_processed_mask.nii.gz",
            f"test{i}_processed_mask_lrflip.nii.gz",
        }


def test_create_subject_dir(tmp_path: Path) -> None:
    """Test that _create_subject_dir creates correct directory structure."""
    sub_id = "test123"
    sub_dir = _create_subject_dir(sub_id, tmp_path)
    assert sub_dir.exists()
    assert sub_dir == tmp_path / "derivatives" / f"sub-{sub_id}"


def test_create_subject_dir_exists(tmp_path: Path) -> None:
    """Test exist_ok=True for _create_subject_dir."""
    sub_id = "test123"
    _create_subject_dir(sub_id, tmp_path)
    _create_subject_dir(sub_id, tmp_path)


def test_save_niftis(
    tmp_path: Path, test_stacks: dict[str, NDArray[np.float64]]
) -> None:
    """Test that _save_niftis saves both standard and flipped images."""
    voxel_sizes = [1.0, 1.0, 1.0]
    image_name = "test_image"

    image_path, flipped_path = _save_niftis(
        test_stacks["image"], voxel_sizes, tmp_path, image_name
    )

    assert image_path.exists() & flipped_path.exists()
    assert image_path.name == f"{image_name}.nii.gz"
    assert flipped_path.name == f"{image_name}_lrflip.nii.gz"


def test_save_niftis_lrflip(
    tmp_path: Path,
    test_stacks: dict[str, NDArray[np.float64]],
) -> None:
    """Test lr-flipped asym images differ from non-flipped ones."""

    voxel_sizes = [1.0, 1.0, 1.0]
    image_name = "test_image"

    image_path, flipped_path = _save_niftis(
        test_stacks["image"], voxel_sizes, tmp_path, image_name
    )

    image = load_any(image_path)
    flipped_image = load_any(flipped_path)

    np.testing.assert_equal(image, np.flip(flipped_image, axis=2))
    with assert_raises(AssertionError):
        np.testing.assert_equal(image, flipped_image)


@pytest.mark.parametrize(
    ["n_sub"],
    [
        pytest.param(1, id="1 subject"),
        pytest.param(2, id="2 subjects"),
    ],
)
def test_process_subject(
    create_raw_test_data: tuple[Path, Path], n_sub: int
) -> None:
    """Test _process_subject creates expected files with expected keys."""
    csv_path, config_path = create_raw_test_data

    # load input csv df with first n subject rows
    input_df = pd.read_csv(csv_path).head(n_sub)

    with open(config_path) as f:
        config_yaml = yaml.safe_load(f)
    config = PreprocConfig.model_validate(config_yaml)

    expected_keys = {"image", "mask", "flipped_image", "flipped_mask"}

    for _, row in input_df.iterrows():
        sub_id = row["subject_id"]
        paths_dict = _process_subject(row, config)

        # Expected naming conventions
        expected_parent = f"sub-{sub_id}"
        expected_filenames = {
            "image": f"{sub_id}_processed.nii.gz",
            "mask": f"{sub_id}_processed_mask.nii.gz",
            "flipped_image": f"{sub_id}_processed_lrflip.nii.gz",
            "flipped_mask": f"{sub_id}_processed_mask_lrflip.nii.gz",
        }

        assert set(paths_dict) == expected_keys

        for key, path in paths_dict.items():
            assert path.exists(), f"{key} not created"
            assert (
                path.parent.name == expected_parent
            ), f"{key} in wrong folder"
            assert (
                path.name == expected_filenames[key]
            ), f"{key} has wrong filename"

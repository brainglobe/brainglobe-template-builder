import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from brainglobe_utils.IO.image import load_any
from numpy.testing import assert_raises
from numpy.typing import NDArray

from brainglobe_template_builder.preprocess import (
    _create_subject_dir,
    _process_subject,
    _save_niftis,
    preprocess,
)
from brainglobe_template_builder.utils.preproc_config import PreprocConfig


@pytest.fixture()
def write_standardised_test_data(
    test_data: list[dict[str, Any]],
    make_tmp_dir,
    write_test_data,
) -> tuple[Path, Path] | Path:
    """Create standardised test data with CSV and config."""

    # Assuming reorientation to ASR has happened
    for test_data_i in test_data:
        test_data_i["origin"] = "ASR"

    return write_test_data(
        dir=make_tmp_dir("standardised"),
        test_data=test_data,
        image_type="nifti",
        csv_name="standardised_data",
        config=True,
    )


@pytest.mark.parametrize(
    ["use", "expected_listdir"],
    [
        pytest.param(["false", "false"], {}, id="false, false"),
        pytest.param(["False", "false"], {}, id="False, false"),
        pytest.param(["false", "true"], {"sub-b"}, id="false, true"),
        pytest.param(["", "false"], {"sub-a"}, id="empty (true), false"),
        pytest.param(["F", "T"], {"sub-a", "sub-b"}, id="F (true), T (true)"),
        pytest.param(["0", "1"], {"sub-a", "sub-b"}, id="0 (true), 1 (true)"),
    ],
)
@pytest.mark.usefixtures("mock_fancylog_datetime")
def test_preprocess_use_input(
    use: list[str],
    expected_listdir: set,
    write_standardised_test_data: tuple[Path, Path],
) -> None:
    """Test exclusion/inclusion based on optional 'use' column values."""
    csv_path, config_path = write_standardised_test_data
    input_df = pd.read_csv(csv_path)
    input_df["use"] = use
    input_df.to_csv(csv_path, index=False)

    preprocessed_dir = csv_path.parents[1] / "preprocessed"

    always_expect = {
        "all_processed_brain_paths.txt",
        "all_processed_mask_paths.txt",
        "template_builder_2025-12-10_15-15-00.log",
    }

    preprocess(csv_path, config_path)
    assert set(os.listdir(preprocessed_dir)) == always_expect.union(
        expected_listdir
    )


@pytest.mark.parametrize(
    "config_type",
    ["config_file", "PreprocConfig object"],
)
@pytest.mark.usefixtures("mock_fancylog_datetime")
def test_preprocess(
    write_standardised_test_data: tuple[Path, Path], config_type: str
) -> None:
    """Test that preprocess creates expected directories and files - both
    with a config yaml file path as input OR a PreprocConfig object."""
    csv_path, config_path = write_standardised_test_data

    config: Path | PreprocConfig
    if config_type == "config_file":
        config = config_path
    else:
        with open(config_path) as f:
            config_yaml = yaml.safe_load(f)
        config = PreprocConfig.model_validate(config_yaml)

    preprocess(csv_path, config)

    preprocessed_dir = csv_path.parents[1] / "preprocessed"
    qc_dir = csv_path.parents[1] / "preprocessed-QC"

    assert preprocessed_dir.exists()
    assert qc_dir.exists()

    assert set(os.listdir(preprocessed_dir)) == {
        "all_processed_brain_paths.txt",
        "all_processed_mask_paths.txt",
        "sub-a",
        "sub-b",
        "template_builder_2025-12-10_15-15-00.log",
    }

    assert set(os.listdir(qc_dir)) == {
        "sub-a-mask-QC-grid.pdf",
        "sub-a-mask-QC-grid.png",
        "sub-b-mask-QC-grid.pdf",
        "sub-b-mask-QC-grid.png",
    }

    for sub_id in ["a", "b"]:
        assert set(os.listdir(preprocessed_dir / f"sub-{sub_id}")) == {
            f"{sub_id}_processed.nii.gz",
            f"{sub_id}_processed_lrflip.nii.gz",
            f"{sub_id}_processed_mask.nii.gz",
            f"{sub_id}_processed_mask_lrflip.nii.gz",
        }


def test_create_subject_dir(tmp_path: Path) -> None:
    """Test that _create_subject_dir creates correct directory structure."""
    sub_id = "a"
    sub_dir = _create_subject_dir(sub_id, tmp_path)
    assert sub_dir.exists()
    assert sub_dir == tmp_path / "preprocessed" / f"sub-{sub_id}"


def test_create_subject_dir_exists(tmp_path: Path) -> None:
    """Test exist_ok=True for _create_subject_dir."""
    sub_id = "a"
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
    write_standardised_test_data: tuple[Path, Path], n_sub: int
) -> None:
    """Test _process_subject creates expected files with expected keys."""
    csv_path, config_path = write_standardised_test_data

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


def test_process_subject_config_padding(
    write_standardised_test_data: tuple[Path, Path],
) -> None:
    """Test whether _process_subject uses padding from config file."""
    csv_path, config_path = write_standardised_test_data
    subject_row = pd.read_csv(csv_path).iloc[0]

    with open(config_path) as f:
        config_yaml = yaml.safe_load(f)

    default_pad = 5
    config_yaml["pad_pixels"] = default_pad
    config_default = PreprocConfig.model_validate(config_yaml)
    paths_default = _process_subject(subject_row, config_default)
    mask_default = load_any(paths_default["mask"])
    image_default = load_any(paths_default["image"])

    config_yaml["pad_pixels"] = 0
    config_nopad = PreprocConfig.model_validate(config_yaml)
    paths_nopad = _process_subject(subject_row, config_nopad)
    mask_nopad = load_any(paths_nopad["mask"])

    assert (
        image_default.shape == mask_default.shape
    ), "Padding should be added to the image as well as the mask."
    expected_shape = tuple(s - default_pad * 2 for s in mask_default.shape)
    assert (
        mask_nopad.shape == expected_shape
    ), f"Mask with no padding should be {default_pad * 2} smaller per dim."


def test_process_subject_config_mask(
    write_standardised_test_data: tuple[Path, Path],
) -> None:
    """Test whether _process_subject uses mask from config file."""
    csv_path, config_path = write_standardised_test_data
    subject_row = pd.read_csv(csv_path).iloc[0]

    with open(config_path) as f:
        config_yaml = yaml.safe_load(f)

    config_default = PreprocConfig.model_validate(config_yaml)
    paths_default = _process_subject(subject_row, config_default)
    mask_default = load_any(paths_default["mask"])

    config_yaml["mask"]["closing_size"] += 5  # increase closing size
    config_changed_mask = PreprocConfig.model_validate(config_yaml)
    paths_changed_mask = _process_subject(subject_row, config_changed_mask)
    mask_changed = load_any(paths_changed_mask["mask"])

    with assert_raises(AssertionError):
        np.testing.assert_equal(mask_changed, mask_default)

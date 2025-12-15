import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fancylog
import numpy as np
import pandas as pd
import pytest
import yaml
from brainglobe_utils.IO.image.save import save_any, save_as_asr_nii
from numpy.typing import NDArray


def _make_stack(
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


def _create_test_images(
    path: Path,
    test_data: list[dict[str, Any]],
    image_type: str = "nifti",
) -> list[dict[str, Any]]:
    """Save test images and add their paths to the returned test data dicts."""

    file_ext = ".nii.gz" if image_type == "nifti" else ".tif"

    for data in test_data:
        subject_dir = path / data["subject_id"]
        subject_dir.mkdir(parents=True, exist_ok=True)

        data["filepath"] = subject_dir / f"{data['subject_id']}{file_ext}"
        stacks = {"image": (data["image"], data["filepath"])}

        if data["mask"] is not None:
            data["mask_filepath"] = (
                subject_dir / f"{data['subject_id']}_mask{file_ext}"
            )
            stacks["mask"] = (data["mask"], data["mask_filepath"])

        for _, (stack, stack_path) in stacks.items():
            if image_type == "nifti":
                save_as_asr_nii(
                    stack,
                    [v * 0.001 for v in data["voxel_size"]],
                    stack_path,
                )
            else:
                save_any(stack, stack_path)

    return test_data


def _create_test_csv(
    path: Path, test_data: list[dict[str, Any]], csv_name: str
) -> Path:
    """Create CSV file and returns its path."""

    data_dict = test_data.copy()
    for data in data_dict:
        data["resolution_0"] = data["voxel_size"][0]
        data["resolution_1"] = data["voxel_size"][1]
        data["resolution_2"] = data["voxel_size"][2]
        for k in ("voxel_size", "image", "mask"):
            data.pop(k)

    csv_path = path / f"{csv_name}.csv"
    pd.DataFrame(data=data_dict).to_csv(csv_path, index=False)
    return csv_path


def _create_test_yaml(path: Path) -> Path:
    """Create YAML config file and returns its path."""
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


@pytest.fixture
def write_test_data() -> Callable:
    """Callable fixture for writing test data files."""

    def _write_test_data(
        dir: Path,
        test_data: list[dict[str, Any]],
        image_type: str,
        csv_name: str,
        config: bool,
    ) -> Path | tuple[Path, Path]:
        """Write test data, return csv path or (csv, config) paths."""
        _create_test_images(dir, test_data, image_type)
        csv_path = _create_test_csv(dir, test_data, csv_name)
        if config:
            config_path = _create_test_yaml(dir.parent)
            return csv_path, config_path
        return csv_path

    return _write_test_data


@pytest.fixture
def mock_fancylog_datetime(mocker):
    """Mock datetime.now for fancylog to 2025-12-10 15:15.

    This allows the log filename timestamp to remain consistent
    for testing.
    """
    mocker.patch("fancylog.fancylog.datetime")
    fancylog.fancylog.datetime.now.return_value = datetime.datetime(
        2025, 12, 10, 15, 15
    )


@pytest.fixture
def test_stacks() -> dict[str, NDArray[np.float64]]:
    """Create symmetric and asymmetric test images and masks."""
    return {
        "image": _make_stack(offset=5),
        "mask": _make_stack(mask=True, offset=5),
    }


@pytest.fixture
def test_data(
    test_stacks: dict[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    """Create test data for two subjects with different voxel sizes."""
    return [
        {
            "subject_id": "a",
            "image": test_stacks["image"],
            "mask": None,
            "voxel_size": [25, 25, 25],
            "origin": "PSL",
        },
        {
            "subject_id": "b",
            "image": test_stacks["image"],
            "mask": None,
            "voxel_size": [10, 10, 10],
            "origin": "LSA",
        },
    ]


@pytest.fixture
def make_tmp_dir(tmp_path: Path) -> Callable:
    """Callable that creates a subdirectory under tmp_path."""

    def _make_subdir(name: str) -> Path:
        subdir = tmp_path / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    return _make_subdir

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from _pytest.logging import LogCaptureFixture
from brainglobe_utils.IO.image.save import save_as_asr_nii
from loguru import logger
from numpy.typing import NDArray


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Override the pytest caplog fixture, so that it will
    work correctly with loguru."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


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
            "mask": None,
            "voxel_size": [25, 25, 25],
            "origin": "PSL",
        },
        {
            "subject_id": "test2",
            "image": test_stacks["image"],
            "mask": None,
            "voxel_size": [10, 10, 10],
            "origin": "LSA",
        },
    ]


def create_test_images(
    path: Path, test_data: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Save test images and add their paths to the returned test data dicts."""
    for data in test_data:
        image_path = path / data["subject_id"] / f"{data['subject_id']}.nii.gz"
        image_path.parent.mkdir()
        voxel_dimensions_in_mm = [v * 0.001 for v in data["voxel_size"]]
        save_as_asr_nii(data["image"], voxel_dimensions_in_mm, image_path)
        data["filepath"] = image_path
    return test_data


def create_test_csv(path: Path, test_data: list[dict[str, Any]]) -> Path:
    """Creates "standardised_data" CSV file and returns its path."""
    for data in test_data:
        data["resolution_0"] = data["voxel_size"][0]
        data["resolution_1"] = data["voxel_size"][1]
        data["resolution_2"] = data["voxel_size"][2]
        data.pop("voxel_size")
        data.pop("image")
    input_csv = pd.DataFrame(data=test_data)
    csv_path = path / "standardised_data.csv"
    input_csv.to_csv(csv_path, index=False)
    return csv_path


def create_test_yaml(path: Path) -> Path:
    """Creates YAML config file and returns its path."""
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
def create_standardised_test_data(
    tmp_path: Path, test_data: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Sets up temp directory with "standardised" test images, CSV,
    and config files."""
    standardised_dir = tmp_path / "standardised"
    standardised_dir.mkdir(parents=True, exist_ok=True)

    test_data = create_test_images(standardised_dir, test_data)
    csv_path = create_test_csv(standardised_dir, test_data)
    config_path = create_test_yaml(tmp_path)
    return csv_path, config_path

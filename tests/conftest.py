from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from _pytest.logging import LogCaptureFixture
from brainglobe_utils.IO.image.save import save_any, save_as_asr_nii
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


def create_test_csv(
    path: Path, test_data: list[dict[str, Any]], csv_name: str
) -> Path:
    """Creates CSV file and returns its path."""

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

    test_data = create_test_images(standardised_dir, test_data, "nifti")
    csv_path = create_test_csv(
        standardised_dir, test_data, "standardised_data"
    )
    config_path = create_test_yaml(tmp_path)
    return csv_path, config_path

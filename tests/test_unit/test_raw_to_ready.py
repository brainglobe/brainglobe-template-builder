from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.preproc.raw_to_ready import raw_to_ready


@pytest.fixture()
def stack():
    """Create a test image stack"""
    stack = np.zeros((50, 50, 50))
    stack[15:36, 15:36, 15:36] = 0.5
    return stack


@pytest.fixture()
def mask():
    """Create a test mask"""
    mask = np.zeros((50, 50, 50))
    mask[10:41, 10:41, 10:41] = 1
    return mask


@pytest.fixture()
def test_data(stack):
    return [
        {
            "subject_id": "sub-1",
            "image": stack,
            "voxel_size": [1, 1, 1],
            "origin": "ASR",
        },
        {
            "subject_id": "sub-2",
            "image": stack,
            "voxel_size": [2, 2, 2],
            "origin": "ASR",
        },
    ]


def create_test_images(path: Path, test_data):
    for data in test_data:
        image_path = path / data["subject_id"] / f"{data['subject_id']}.nii.gz"
        image_path.parent.mkdir()
        save_as_asr_nii(data["image"], data["voxel_size"], image_path)
        data["source_filepath"] = image_path
    return test_data


def create_test_csv(path, test_data):
    for data in test_data:
        data["resolution_z"] = data["voxel_size"][0]
        data["resolution_y"] = data["voxel_size"][1]
        data["resolution_x"] = data["voxel_size"][2]
        data.pop("voxel_size")
    input_csv = pd.DataFrame(data=test_data)
    csv_path = path / "raw_data.csv"
    input_csv.to_csv(csv_path, index=False)
    return csv_path


def create_test_yaml(path):
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
def create_test_data(tmp_path, test_data):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    test_data = create_test_images(raw_dir, test_data)
    csv_path = create_test_csv(raw_dir, test_data)
    config_path = create_test_yaml(raw_dir)
    return csv_path, config_path


def test_simple(create_test_data):
    csv_path, config_path = create_test_data
    raw_to_ready(csv_path, config_path)
    assert (create_test_data[0].parents[0] / "derivatives").exists()

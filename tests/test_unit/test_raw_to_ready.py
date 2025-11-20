import numpy as np
import pandas as pd
import pytest
import yaml
from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_template_builder.preproc.raw_to_ready import raw_to_ready


@pytest.fixture()
def stack():
    """Create a test image stack"""
    stack = np.ones((50, 50, 50))
    stack[15:36, 15:36, 15:36] = 0.5
    return stack


@pytest.fixture()
def mask():
    """Create a test mask"""
    mask = np.zeros((50, 50, 50))
    mask[10:41, 10:41, 10:41] = 1
    return mask


@pytest.fixture()
def paths_test_data(tmp_path, stack):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    subjects = ["sub-1", "sub-2"]
    voxel_size = [1, 1, 1]
    file_paths = []
    for sub in subjects:
        subject_dir = raw_dir / sub
        subject_dir.mkdir()

        image_path = subject_dir / f"{sub}.nii.gz"
        save_as_asr_nii(stack, voxel_size, image_path)
        file_paths.append(image_path)

    input_csv = pd.DataFrame(
        data={
            "subject_id": subjects,
            "resolution_z": [1, 1],
            "resolution_y": [1, 1],
            "resolution_x": [1, 1],
            "origin": ["ASR", "ASR"],
            "source_filepath": file_paths,
        }
    )
    csv_path = raw_dir / "raw_data.csv"
    input_csv.to_csv(csv_path, index=False)

    yaml_dict = {
        "output_dir": str(tmp_path.resolve()),
        "mask": {
            "gaussian_sigma": 3,
            "threshold_method": "triangle",
            "closing_size": 5,
            "erode_size": 0,
        },
        "pad_pixels": 5,
    }

    config_path = raw_dir / "config.yml"
    with open(config_path, "w") as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False)

    return csv_path, config_path


def test_simple(paths_test_data):
    raw_to_ready(*paths_test_data)

    assert (paths_test_data[0].parents[1] / "derivatives").exists()

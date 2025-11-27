from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from brainglobe_space import AnatomicalSpace
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
    stack[10:30, 10:30, 10:30] = 0.5
    return stack


@pytest.fixture()
def mask() -> NDArray[np.float64]:
    """Create 50x50x50 binary mask with off-centre object (value = 1).

    The object is off-centre, so we can identify the effects of
    re-orientation to ASR.
    """
    mask = np.zeros((50, 50, 50))
    mask[5:35, 5:35, 5:35] = 1
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
def test_data(stack: NDArray[np.float64]) -> list[dict[str, Any]]:
    """Creates test data for two images with LSA and PSL orientation."""
    return [
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


@pytest.fixture()
def source_csv_no_masks(
    source_dir: Path, test_data: list[dict[str, Any]]
) -> Path:
    """Create test data for two subjects - neither of which
    have masks."""

    return write_test_data(source_dir, test_data)


@pytest.fixture()
def source_csv_with_masks(
    source_dir: Path,
    test_data: list[dict[str, Any]],
    mask: NDArray[np.float64],
) -> Path:
    """Create test data for two subjects, one with a mask and
    one without."""

    test_data[0]["mask"] = mask
    test_data[1]["mask"] = None

    return write_test_data(source_dir, test_data)


@pytest.fixture()
def source_csv_single_image_with_mask(
    source_dir: Path,
    test_data: list[dict[str, Any]],
    mask: NDArray[np.float64],
) -> Path:
    """Create test data for a single subject with a
    corresponding mask."""

    single_image = test_data[1]
    single_image["mask"] = mask

    return write_test_data(source_dir, [single_image])


@pytest.fixture()
def source_csv_with_use(
    source_dir: Path, test_data: list[dict[str, Any]]
) -> Path:
    """Create test data for two subjects - one with use=False,
    and the other with use=True."""

    test_data[0]["use"] = False
    test_data[1]["use"] = True

    return write_test_data(source_dir, test_data)


@pytest.fixture()
def source_csv_anisotropic(
    source_dir: Path, stack: NDArray[np.float64]
) -> Path:
    """Create test data for a single subject with anisotropic
    resolution."""

    return write_test_data(
        source_dir,
        [
            {
                "image": stack,
                "mask": None,
                "subject_id": "a",
                "resolution_z": 25,
                "resolution_y": 30,
                "resolution_x": 40,
                "origin": "PSL",
            }
        ],
    )


@pytest.mark.parametrize(
    (
        "source_csv",
        "expected_subject_ids",
        "expected_image_paths",
        "expected_mask_paths",
    ),
    [
        pytest.param(
            "source_csv_no_masks",
            ["a", "b"],
            [
                "raw/sub-a/sub-a_res-50x50x50um_origin-asr.nii.gz",
                "raw/sub-b/sub-b_res-50x50x50um_origin-asr.nii.gz",
            ],
            [],
            id="no mask column",
        ),
        pytest.param(
            "source_csv_with_masks",
            ["a", "b"],
            [
                "raw/sub-a/sub-a_res-50x50x50um_origin-asr.nii.gz",
                "raw/sub-b/sub-b_res-50x50x50um_origin-asr.nii.gz",
            ],
            ["raw/sub-a/sub-a_res-50x50x50um_mask_origin-asr.nii.gz", np.nan],
            id="with mask column",
        ),
    ],
)
def test_source_to_raw_filepaths(
    request,
    source_csv,
    expected_subject_ids,
    expected_image_paths,
    expected_mask_paths,
):
    """Test source to raw creates all the correct files, in the right
    directory structure."""

    source_csv_path = request.getfixturevalue(source_csv)
    output_dir = source_csv_path.parents[1]
    output_vox_size = 50
    source_to_raw(source_csv_path, output_dir, output_vox_size)

    # Should create a raw dir, with one directory per subject id
    raw_dir = output_dir / "raw"
    assert raw_dir.exists()
    sub_dirs = [path.stem for path in raw_dir.iterdir() if path.is_dir()]
    assert sorted(sub_dirs) == sorted(
        [f"sub-{id}" for id in expected_subject_ids]
    )

    # Each subject directory should contain an image (+ optional mask)
    # and QC plots
    expected_image_paths = [output_dir / path for path in expected_image_paths]
    expected_mask_paths = [
        output_dir / path if not pd.isna(path) else path
        for path in expected_mask_paths
    ]

    for i, subject_id in enumerate(expected_subject_ids):
        subject_dir = raw_dir / f"sub-{subject_id}"
        subject_files = list(subject_dir.iterdir())

        expected_files = [expected_image_paths[i]]
        if expected_mask_paths and not pd.isna(expected_mask_paths[i]):
            expected_files.append(expected_mask_paths[i])

        # Add QC plots for every image and mask
        plot_paths = []
        for file_path in expected_files:
            plot_paths.append(
                Path(str(file_path).removesuffix(".nii.gz") + "-QC.png")
            )
            plot_paths.append(
                Path(str(file_path).removesuffix(".nii.gz") + "-QC.pdf")
            )
        expected_files.extend(plot_paths)

        assert sorted(subject_files) == sorted(expected_files)

    # Check output csv exists with correct metadata
    output_csv_path = raw_dir / "raw_images.csv"
    assert output_csv_path.exists()

    output_csv = pd.read_csv(output_csv_path)
    expected_output_csv = pd.DataFrame(
        data={
            "subject_id": expected_subject_ids,
            "resolution_z": output_vox_size,
            "resolution_y": output_vox_size,
            "resolution_x": output_vox_size,
            "origin": "ASR",
            "source_filepath": np.array(expected_image_paths).astype(str),
        }
    )
    if expected_mask_paths:
        mask_paths_str = [
            str(path) if not pd.isna(path) else path
            for path in expected_mask_paths
        ]
        expected_output_csv["mask_filepath"] = mask_paths_str

    pd.testing.assert_frame_equal(output_csv, expected_output_csv)


def test_source_to_raw_with_use(source_csv_with_use):
    """Test source_to_raw excludes subjects with use=False."""

    output_dir = source_csv_with_use.parents[1]
    source_to_raw(source_csv_with_use, output_dir, 50)

    raw_dir = output_dir / "raw"
    assert raw_dir.exists()
    assert not (raw_dir / "sub-a").exists()  # subject a has use=False
    assert (raw_dir / "sub-b").exists()  # subject b has use=True

    output_csv_path = output_dir / "raw" / "raw_images.csv"
    assert output_csv_path.exists()

    output_csv = pd.read_csv(output_csv_path)
    assert len(output_csv) == 1
    assert output_csv.subject_id.iloc[0] == "b"


def test_source_to_raw_anisotropic(source_csv_anisotropic):
    """Test source_to_raw errors when anisotropic data is provided with no
    output_vox_size.
    """

    with pytest.raises(
        ValueError,
        match=r"Subject id: a has anisotropic voxel size: \[40, 30, 25\]",
    ):
        source_to_raw(
            source_csv_anisotropic, source_csv_anisotropic.parents[1]
        )


def test_source_to_raw_reorientation(
    source_csv_single_image_with_mask, stack, mask
):
    """Test source_to_raw re-orients images and masks to ASR."""

    output_dir = source_csv_single_image_with_mask.parents[1]
    source_to_raw(source_csv_single_image_with_mask, output_dir)

    subject_dir = output_dir / "raw" / "sub-b"
    image_path = subject_dir / "sub-b_res-10x10x10um_origin-asr.nii.gz"
    mask_path = subject_dir / "sub-b_res-10x10x10um_mask_origin-asr.nii.gz"

    for output_path, source_image in zip(
        [image_path, mask_path], [stack, mask]
    ):
        # No downsampling was specified, so input size / res should
        # match output
        image = load_nii(output_path, as_array=False)
        assert image.header.get_zooms() == (
            0.01,
            0.01,
            0.01,
        )  # saved to nii as mm
        assert image.shape == (50, 50, 50)

        # Output should match re-orientation of source image to ASR
        expected_image = AnatomicalSpace("LSA").map_stack_to(
            "ASR", source_image
        )
        np.testing.assert_equal(image.get_fdata(), expected_image)


def test_source_to_raw_downsampling(source_csv_single_image_with_mask):
    """Test source_to_raw downsamples images + masks to the correct size."""

    output_dir = source_csv_single_image_with_mask.parents[1]
    output_vox_size = 20
    source_to_raw(source_csv_single_image_with_mask, output_dir)

    subject_dir = output_dir / "raw" / "sub-b"
    image_path = subject_dir / "sub-b_res-10x10x10um_origin-asr.nii.gz"
    mask_path = subject_dir / "sub-b_res-10x10x10um_mask_origin-asr.nii.gz"

    output_vox_sizes_mm = (
        output_vox_size * 0.001,
        output_vox_size * 0.001,
        output_vox_size * 0.001,
    )

    for output_path in [image_path, mask_path]:
        # Output voxel size (20) is double input (10) - so image should
        # be half the size
        image = load_nii(output_path, as_array=False)
        assert image.header.get_zooms() == output_vox_sizes_mm
        assert image.shape == (25, 25, 25)

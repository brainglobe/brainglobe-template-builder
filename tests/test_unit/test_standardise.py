import datetime
from pathlib import Path
from typing import Any

import fancylog
import numpy as np
import pandas as pd
import pytest
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image.load import load_nii
from brainglobe_utils.IO.image.save import save_any
from numpy.typing import NDArray

from brainglobe_template_builder.preproc.standardise import standardise


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
        subject_dir.mkdir()

        image_path = subject_dir / f"{data['subject_id']}.tiff"
        save_any(data["image"], image_path)
        data["filepath"] = image_path

        if data["mask"] is not None:
            mask_path = subject_dir / f"{data['subject_id']}_mask.tiff"
            save_any(data["mask"], mask_path)
            data["mask_filepath"] = mask_path

    return test_data


def create_test_csv(path: Path, test_data: list[dict[str, Any]]) -> Path:
    """Creates "source_data" CSV file and returns its path."""

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
    """Write test data to source_dir, and return the path to
    the summary csv."""
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
            "resolution_0": 25,
            "resolution_1": 25,
            "resolution_2": 25,
            "origin": "PSL",
        },
        {
            "image": stack,
            "mask": None,
            "subject_id": "b",
            "resolution_0": 10,
            "resolution_1": 10,
            "resolution_2": 10,
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
def source_csv_anisotropic_with_mask(
    source_dir: Path, stack: NDArray[np.float64], mask: NDArray[np.float64]
) -> Path:
    """Create test data for a single subject with anisotropic
    resolution and a corresponding mask."""

    return write_test_data(
        source_dir,
        [
            {
                "image": stack,
                "mask": mask,
                "subject_id": "b",
                "resolution_0": 10,
                "resolution_1": 4,
                "resolution_2": 2,
                "origin": "ASR",
            }
        ],
    )


@pytest.mark.parametrize(
    "source_csv,expected_standardised_paths,expected_qc_paths",
    [
        pytest.param(
            "source_csv_no_masks",
            [
                "standardised/sub-a",
                "standardised/sub-b",
                "standardised/standardised_images.csv",
                "standardised/template_builder_2025-12-10_15-15-00.log",
                "standardised/sub-a/sub-a_res-50x50x50um_origin-asr.nii.gz",
                "standardised/sub-b/sub-b_res-50x50x50um_origin-asr.nii.gz",
            ],
            [
                "standardised-QC/sub-a-QC-orthographic.png",
                "standardised-QC/sub-a-QC-orthographic.pdf",
                "standardised-QC/sub-b-QC-orthographic.png",
                "standardised-QC/sub-b-QC-orthographic.pdf",
            ],
            id="no mask column",
        ),
        pytest.param(
            "source_csv_with_masks",
            [
                "standardised/sub-a",
                "standardised/sub-b",
                "standardised/standardised_images.csv",
                "standardised/template_builder_2025-12-10_15-15-00.log",
                # subject a image + mask
                "standardised/sub-a/sub-a_res-50x50x50um_origin-asr.nii.gz",
                "standardised/sub-a/sub-a_res-50x50x50um_mask_origin-asr.nii.gz",
                # subject b image (no mask provided)
                "standardised/sub-b/sub-b_res-50x50x50um_origin-asr.nii.gz",
            ],
            [
                # subject a QC plots for image + mask
                "standardised-QC/sub-a-QC-orthographic.png",
                "standardised-QC/sub-a-QC-orthographic.pdf",
                "standardised-QC/sub-a-mask-QC-orthographic.png",
                "standardised-QC/sub-a-mask-QC-orthographic.pdf",
                # subject b QC plot for image (no mask provided)
                "standardised-QC/sub-b-QC-orthographic.png",
                "standardised-QC/sub-b-QC-orthographic.pdf",
            ],
            id="with mask column",
        ),
    ],
)
def test_standardise_filepaths(
    request, mocker, source_csv, expected_standardised_paths, expected_qc_paths
):
    """Test standardise creates all the correct files, in the right
    directory structure."""

    source_csv_path = request.getfixturevalue(source_csv)
    output_dir = source_csv_path.parents[1]
    output_vox_size = 50

    # Mock datetime.now for fancylog - so the log filename timestamp
    # remains consistent
    mocker.patch("fancylog.fancylog.datetime")
    fancylog.fancylog.datetime.now.return_value = datetime.datetime(
        2025, 12, 10, 15, 15
    )
    standardise(source_csv_path, output_dir, output_vox_size)

    # Check correct files / directory structure created
    standardised_dir = output_dir / "standardised"
    qc_dir = output_dir / "standardised-QC"

    for dir_path, expected_paths in zip(
        [standardised_dir, qc_dir],
        [expected_standardised_paths, expected_qc_paths],
    ):
        assert dir_path.exists()

        created_paths = list(dir_path.glob("**/*"))
        expected_paths = [output_dir / file for file in expected_paths]

        assert sorted(created_paths) == sorted(expected_paths)


@pytest.mark.parametrize(
    "source_csv,expected_image_paths,expected_mask_paths",
    [
        pytest.param(
            "source_csv_no_masks",
            [
                "standardised/sub-a/sub-a_res-50x50x50um_origin-asr.nii.gz",
                "standardised/sub-b/sub-b_res-50x50x50um_origin-asr.nii.gz",
            ],
            [],
            id="no mask column",
        ),
        pytest.param(
            "source_csv_with_masks",
            [
                "standardised/sub-a/sub-a_res-50x50x50um_origin-asr.nii.gz",
                "standardised/sub-b/sub-b_res-50x50x50um_origin-asr.nii.gz",
            ],
            [
                "standardised/sub-a/sub-a_res-50x50x50um_mask_origin-asr.nii.gz",
                np.nan,
            ],
            id="with mask column",
        ),
    ],
)
def test_standardise_output_csv(
    request, source_csv, expected_image_paths, expected_mask_paths
):
    """Test standardise creates an output csv with the correct metadata."""

    source_csv_path = request.getfixturevalue(source_csv)
    output_dir = source_csv_path.parents[1]
    output_vox_size = 50
    standardise(source_csv_path, output_dir, output_vox_size)

    # Check output csv contains correct paths
    output_csv_path = output_dir / "standardised" / "standardised_images.csv"
    output_csv = pd.read_csv(output_csv_path)

    expected_image_paths = [
        str(output_dir / path) for path in expected_image_paths
    ]
    expected_mask_paths = [
        str(output_dir / path) if not pd.isna(path) else path
        for path in expected_mask_paths
    ]

    expected_output_csv = pd.DataFrame(
        data={
            "subject_id": ["a", "b"],
            "resolution_0": output_vox_size,
            "resolution_1": output_vox_size,
            "resolution_2": output_vox_size,
            "origin": "ASR",
            "filepath": expected_image_paths,
        }
    )
    if len(expected_mask_paths) > 0:
        expected_output_csv["mask_filepath"] = expected_mask_paths

    pd.testing.assert_frame_equal(output_csv, expected_output_csv)


def test_standardise_with_use(source_csv_with_use):
    """Test standardise excludes subjects with use=False."""

    output_dir = source_csv_with_use.parents[1]
    standardise(source_csv_with_use, output_dir)

    standardised_dir = output_dir / "standardised"
    assert standardised_dir.exists()
    assert not (standardised_dir / "sub-a").exists()  # subject a has use=False
    assert (standardised_dir / "sub-b").exists()  # subject b has use=True

    output_csv_path = standardised_dir / "standardised_images.csv"
    assert output_csv_path.exists()

    output_csv = pd.read_csv(output_csv_path)
    assert len(output_csv) == 1
    assert output_csv.subject_id.iloc[0] == "b"


def test_standardise_anisotropic(source_csv_anisotropic_with_mask):
    """Test standardise errors when anisotropic data is provided with no
    output_vox_size.
    """

    with pytest.raises(
        ValueError,
        match=r"Subject id: b has anisotropic voxel size: \[10, 4, 2\]",
    ):
        standardise(
            source_csv_anisotropic_with_mask,
            source_csv_anisotropic_with_mask.parents[1],
        )


def test_standardise_reorientation(
    source_csv_single_image_with_mask, stack, mask
):
    """Test standardise re-orients images and masks to ASR."""

    output_dir = source_csv_single_image_with_mask.parents[1]
    standardise(source_csv_single_image_with_mask, output_dir)

    subject_dir = output_dir / "standardised" / "sub-b"
    image_path = subject_dir / "sub-b_res-10x10x10um_origin-asr.nii.gz"
    mask_path = subject_dir / "sub-b_res-10x10x10um_mask_origin-asr.nii.gz"

    for output_path, source_image in zip(
        [image_path, mask_path], [stack, mask]
    ):
        # No downsampling was specified, so input size / res should
        # match output
        image = load_nii(output_path, as_array=False)

        expected_zooms = (
            0.01,
            0.01,
            0.01,
        )  # saved to nii as mm

        np.testing.assert_allclose(image.header.get_zooms(), expected_zooms)
        assert image.shape == (50, 50, 50)

        # Output should match re-orientation of source image to ASR
        expected_image = AnatomicalSpace("LSA").map_stack_to(
            "ASR", source_image
        )
        np.testing.assert_equal(image.get_fdata(), expected_image)


@pytest.mark.parametrize(
    "source_csv,expected_output_size",
    [
        pytest.param(
            "source_csv_single_image_with_mask",
            (25, 25, 25),
            id="isotropic input - scale by 0.5, 0.5, 0.5",
        ),
        pytest.param(
            "source_csv_anisotropic_with_mask",
            (25, 10, 5),
            id="anisotropic input - scale by 0.5, 0.2, 0.1",
        ),
    ],
)
def test_standardise_downsampling(request, source_csv, expected_output_size):
    """Test standardise downsamples images + masks to the correct size."""

    source_csv_path = request.getfixturevalue(source_csv)
    output_dir = source_csv_path.parents[1]
    output_vox_size = 20
    standardise(source_csv_path, output_dir, output_vox_size)

    subject_dir = output_dir / "standardised" / "sub-b"
    image_path = subject_dir / "sub-b_res-20x20x20um_origin-asr.nii.gz"
    mask_path = subject_dir / "sub-b_res-20x20x20um_mask_origin-asr.nii.gz"

    output_vox_sizes_mm = (
        output_vox_size * 0.001,
        output_vox_size * 0.001,
        output_vox_size * 0.001,
    )

    for output_path in [image_path, mask_path]:
        image = load_nii(output_path, as_array=False)
        np.testing.assert_allclose(
            image.header.get_zooms(), output_vox_sizes_mm
        )
        assert image.shape == expected_output_size

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.IO.image.load import load_nii
from numpy._typing._array_like import NDArray

from brainglobe_template_builder.standardise import standardise


@pytest.fixture
def source_dir(
    make_tmp_dir: Callable[[str], Path],
) -> tuple[Path, Path] | Path:
    """Create a temporary 'source' directory."""
    return make_tmp_dir("source")


@pytest.fixture
def source_data_kwargs(make_tmp_dir: Callable, test_data: list[dict]) -> dict:
    return {
        "image_type": "tif",
        "csv_name": "source_data",
        "config": False,
        "dir": make_tmp_dir("source"),
        "test_data": test_data,
    }


@pytest.fixture
def source_csv_no_masks(
    write_test_data: Callable, source_data_kwargs: dict
) -> tuple[Path, Path] | Path:
    """Create source test data with CSV and config."""
    return write_test_data(**source_data_kwargs)


@pytest.fixture
def source_csv_with_masks(
    source_data_kwargs: dict[str, Any],
    test_stacks: dict[str, NDArray[np.float64]],
    write_test_data: Callable,
) -> tuple[Path, Path] | Path:
    """Create test data for two subjects, one with a mask and
    one without."""

    source_data_kwargs["test_data"][0]["mask"] = test_stacks["mask"]
    source_data_kwargs["test_data"][1]["mask"] = None

    return write_test_data(**source_data_kwargs)


@pytest.fixture
def source_csv_single_image_with_mask(
    source_data_kwargs: dict[str, Any],
    test_stacks: dict[str, NDArray[np.float64]],
    write_test_data: Callable,
) -> tuple[Path, Path] | Path:
    """Create test data for a single subject with a
    corresponding mask."""

    source_data_kwargs["test_data"] = [source_data_kwargs["test_data"][1]]
    source_data_kwargs["test_data"][0]["mask"] = test_stacks["mask"]

    return write_test_data(**source_data_kwargs)


@pytest.fixture
def source_csv_with_use(
    source_data_kwargs: dict[str, Any], write_test_data: Callable
) -> tuple[Path, Path] | Path:
    """Create test data for two subjects - one with use=False,
    and the other with use=True."""

    source_data_kwargs["test_data"][0]["use"] = False
    source_data_kwargs["test_data"][1]["use"] = True

    return write_test_data(**source_data_kwargs)


@pytest.fixture
def source_csv_anisotropic_with_mask(
    source_data_kwargs: dict[str, Any],
    test_stacks: dict[str, NDArray[np.float64]],
    write_test_data: Callable,
) -> tuple[Path, Path] | Path:
    """Create test data for a single subject with anisotropic
    resolution and a corresponding mask."""

    source_data_kwargs["test_data"] = [
        {
            "image": test_stacks["image"],
            "mask": test_stacks["mask"],
            "subject_id": "b",
            "voxel_size": [10, 4, 2],
            "origin": "ASR",
        }
    ]
    return write_test_data(**source_data_kwargs)


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
@pytest.mark.usefixtures("mock_fancylog_datetime")
def test_standardise_filepaths(
    request, source_csv, expected_standardised_paths, expected_qc_paths
):
    """Test standardise creates all the correct files, in the right
    directory structure."""

    source_csv_path = request.getfixturevalue(source_csv)
    output_dir = source_csv_path.parents[1]
    output_vox_size = 50
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

    pd.testing.assert_frame_equal(
        output_csv, expected_output_csv, check_like=True
    )


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
    source_csv_single_image_with_mask, test_stacks
):
    """Test standardise re-orients images and masks to ASR."""

    output_dir = source_csv_single_image_with_mask.parents[1]
    standardise(source_csv_single_image_with_mask, output_dir)

    subject_dir = output_dir / "standardised" / "sub-b"
    image_path = subject_dir / "sub-b_res-10x10x10um_origin-asr.nii.gz"
    mask_path = subject_dir / "sub-b_res-10x10x10um_mask_origin-asr.nii.gz"

    for output_path, source_image in zip(
        [image_path, mask_path], [test_stacks["image"], test_stacks["mask"]]
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

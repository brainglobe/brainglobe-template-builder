import warnings

import numpy as np
import pytest

from brainglobe_template_builder.utils.alignment import (
    MidplaneAligner,
    MidplaneEstimator,
)


@pytest.fixture
def test_data():
    """Create asymmetric test data with stack, mask, and rotated stack."""
    stack = np.zeros((50, 50, 50), dtype=np.float32)
    stack[10:30, 15:40, 20:45] = 0.5
    mask = (stack > 0).astype(bool)
    points = MidplaneEstimator(mask, symmetry_axis="x").get_points()

    return {
        "stack": stack,
        "mask": mask,
        "points": points,
    }


def test_midplane_estimator_validate_symmetry_axis(test_data):
    """Test validation of axis label upon MidplaneEstimator creation."""
    invalid_label = "b"
    with pytest.raises(ValueError, match="Symmetry axis must be one of"):
        MidplaneEstimator(mask=test_data["mask"], symmetry_axis=invalid_label)


def test_midplane_aligner_validate_symmetry_axis(test_data):
    """Test validation of axis label upon MidplaneAligner creation."""
    invalid_label = "b"
    with pytest.raises(ValueError, match="Symmetry axis must be one of"):
        MidplaneAligner(
            image=test_data["stack"],
            points=test_data["points"],
            symmetry_axis=invalid_label,
        )


def test_midplane_estimator_validate_2Dmask(test_data):
    """Test MidplaneEstimator mask validation for invalid 2D mask."""
    mask2D = test_data["mask"][:, :, 0]
    with pytest.raises(ValueError, match="Mask must be 3D"):
        MidplaneEstimator(mask=mask2D, symmetry_axis="x")


def test_midplane_estimator_validate_2Dimage(test_data):
    """Test MidplaneAligner image validation for invalid 2D image."""
    image2D = test_data["stack"][:, :, 0]
    with pytest.raises(ValueError, match="Image must be 3D"):
        MidplaneAligner(
            image=image2D, points=test_data["points"], symmetry_axis="x"
        )


@pytest.mark.skip(reason="Handling of non-binary masks TBD (issue #167)")
@pytest.mark.parametrize(
    "dtype, values, indexes",
    [
        pytest.param(
            np.float32,
            [0.5],
            [[10, 15, 20]],
            id="float mask",
        ),
        pytest.param(
            np.uint8,
            [1, 2],
            [[10, 15, 20], [30, 40, 45]],
            id="uint8 mask with 0, 1, and 2 values",
        ),
    ],
)
def test_midplane_estimator_validate_nonbinary_mask(dtype, values, indexes):
    """Test MidplaneEstimator mask validation for non-binary masks."""

    non_binary_mask = np.zeros((50, 50, 50), dtype=dtype)
    for i, value in enumerate(values):
        start = indexes[i]
        end = [index + 5 for index in indexes[i]]
        non_binary_mask[
            start[0] : end[0], start[1] : end[1], start[2] : end[2]
        ] = value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        midplane_estimator = MidplaneEstimator(
            mask=non_binary_mask, symmetry_axis="x"
        )
    assert len(caught_warnings) == 1
    assert "Converting to boolean" in str(caught_warnings[0].message)
    assert midplane_estimator.mask.dtype == bool


@pytest.mark.skip(reason="Handling of non-boolean masks TBD (issue #167)")
def test_midplane_estimator_validate_bool_mask(test_data):
    """Test mask validation of MidplaneEstimator object for boolean masks."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        MidplaneEstimator(mask=test_data["mask"], symmetry_axis="x")
    assert len(caught_warnings) == 0


@pytest.mark.parametrize(
    "points_modification",
    [
        pytest.param(
            lambda p: p.transpose(),
            id="transposed (3, 9)",
        ),
        pytest.param(
            lambda p: p[:, 0],
            id="collapsed to 1D (9,)",
        ),
    ],
)
def test_midplane_estimator_validate_points_shape(
    test_data, points_modification
):
    """Test MidplaneAligner image validation of points."""
    points = points_modification(test_data["points"])
    with pytest.raises(ValueError, match="Points must be an array of shape"):
        MidplaneAligner(
            image=test_data["stack"], points=points, symmetry_axis="x"
        )

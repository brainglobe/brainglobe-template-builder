import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from brainglobe_template_builder.napari.align_widget import AlignMidplane
from brainglobe_template_builder.utils.alignment import (
    MidplaneEstimator,
)


@pytest.fixture
def test_data():
    """Create asymmetric test data with stack, mask, and rotated stack."""
    stack = np.zeros((50, 50, 50), dtype=np.float32)
    stack[10:30, 15:40, 20:45] = 0.5
    mask = (stack > 0).astype(bool)

    return {
        "stack": stack,
        "mask": mask,
    }


@pytest.fixture
def align_widget(make_napari_viewer, test_data):
    """Creates a napari viewer with the AlignMidplane widget
    docked and a test stack layer added."""
    viewer = make_napari_viewer()
    viewer.add_image(test_data["stack"], name="test_stack")
    viewer.add_labels(test_data["mask"], name="test_mask")
    align_widget = AlignMidplane(viewer)
    viewer.window.add_dock_widget(align_widget)
    return align_widget


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param("x", id="x axis"),
        pytest.param("y", id="y axis"),
        pytest.param("z", id="z axis"),
    ],
)
def test_estimate_points(align_widget, test_data, axis):
    """Test that estimate points creates a Points layer with correct data."""
    align_widget.select_axis_dropdown.setCurrentText(axis)
    align_widget._on_estimate_button_click()
    points_layer = align_widget.viewer.layers[-1]  # Last added layer

    # get estimated points for specified symmetry axis
    midplane_estimate = MidplaneEstimator(
        test_data["mask"], symmetry_axis=axis
    )
    expected_points = midplane_estimate.get_points()
    np.testing.assert_array_equal(points_layer.data, expected_points)


@pytest.mark.parametrize(
    "symmetry_axis",
    [
        pytest.param("z", id="z axis"),
        pytest.param("y", id="y axis"),
        pytest.param("x", id="x axis"),
    ],
)
def test_estimate_points_symmetry_axis(align_widget, symmetry_axis):
    """Check whether the coordinates along the symmetry axis stay constant.

    For a given symmetry axis, all points in the estimated midplane should
    have the same coordinate value along that axis."""
    axis_dict = {"z": 0, "y": 1, "x": 2}
    align_widget.select_axis_dropdown.setCurrentText(symmetry_axis)
    align_widget._on_estimate_button_click()
    points = align_widget.viewer.layers[-1].data
    sym_axis_coordinates = points[:, axis_dict[symmetry_axis]]
    assert np.all(sym_axis_coordinates == sym_axis_coordinates[0])


@pytest.mark.parametrize(
    "alignment_needed",
    [
        pytest.param(True, id="alignment needed"),
        pytest.param(
            False,
            id="already aligned",
        ),
    ],
)
def test_align_midplane(align_widget, alignment_needed):
    """Test that midplane can be aligned without raising an exception.

    Test scenario when rotation is needed (by adding offset to the by default
    perfectly aligned points) and when it is not."""

    viewer = align_widget.viewer
    mask = viewer.layers["test_mask"].data
    points = MidplaneEstimator(mask, symmetry_axis="x").get_points()

    if alignment_needed:
        points[0] += [1, 3, 2]

    viewer.add_points(points, name="test_points-x")
    align_widget.refresh_dropdowns()

    try:
        align_widget._on_align_button_click()
    except Exception as e:
        pytest.fail(f'_on_align_button_click raised "{e}"')


def test_align_save_transform(align_widget, tmp_path):
    """Test saving the alignment transform matrix through user interaction.

    Verifies that the transform matrix can be saved when the user
    selects a save path via QFileDialog (which is mocked in this test).
    The test checks whether a file is created and that the saved
    file contains the expected transformation matrix.
    """

    viewer = align_widget.viewer
    mask = viewer.layers["test_mask"].data
    points = MidplaneEstimator(mask, symmetry_axis="x").get_points()
    viewer.add_points(points, name="test_points-x")
    align_widget.refresh_dropdowns()
    align_widget._on_align_button_click()

    transform_filepath = str(tmp_path / "transform")
    with patch(
        "brainglobe_template_builder.napari.align_widget.QFileDialog"
    ) as mock_qfile_dialog:
        mock_qfile_dialog.return_value.exec_.return_value = True
        mock_qfile_dialog.return_value.selectedFiles.return_value = [
            transform_filepath
        ]
        align_widget._on_save_transform_click()

    assert Path(transform_filepath).exists(), "Transform file was not created."
    saved_transform = np.loadtxt(transform_filepath)
    expected_transform = align_widget.aligner.transform

    assert np.allclose(
        saved_transform,
        expected_transform,
    )


def test_midplane_estimator_validate_symmetry_axis(test_data):
    """Test validation of axis label upon MidplaneEstimator creation."""
    invalid_label = "b"
    with pytest.raises(ValueError, match="Symmetry axis must be one of"):
        MidplaneEstimator(mask=test_data["mask"], symmetry_axis=invalid_label)


def test_midplane_estimator_validate_2Dmask():
    """Test MidplaneEstimator mask validation for invalid 2D mask."""
    mask2D = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match="Mask must be 3D"):
        MidplaneEstimator(mask=mask2D, symmetry_axis="x")


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

import numpy as np
import pytest
from scipy.ndimage import rotate

from brainglobe_template_builder.napari.align_widget import AlignMidplane
from brainglobe_template_builder.utils.alignment import (
    MidplaneEstimator,
)


@pytest.fixture
def test_data():
    """Create asymmetric test data with stack, mask, and rotated stack."""
    stack = np.zeros((50, 50, 50), dtype=np.float32)
    stack[10:30, 15:40, 20:45] = 0.5
    mask = (stack > 0).astype(np.uint8)

    rotated = stack.copy()
    for axes in [(1, 2), (0, 2), (0, 1)]:  # rotate around all 3 mid planes
        rotated = rotate(rotated, angle=15, axes=axes, order=0, reshape=True)
    mask_rotated = (rotated > 0).astype(np.uint8)

    return {
        "stack": stack,
        "mask": mask,
        "stack_rotated": rotated,
        "mask_rotated": mask_rotated,
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


@pytest.mark.xfail(reason="bug (remove marker after PR #161 is merged)")
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
    "rotation_needed",
    [
        pytest.param(True, id="rotation needed"),
        pytest.param(
            False,
            id="no rotation needed",
            marks=pytest.mark.xfail(
                reason="bug (remove marker once issue #155 is resolved)"
            ),
        ),
    ],
)
def test_align_midplane(make_napari_viewer, test_data, rotation_needed):
    """Test that midplane can be aligned without raising an exception.

    Test scenario when rotation is needed (by adding offset to the by default
    perfectly aligned points) and when it is not."""

    viewer = make_napari_viewer()
    viewer.add_image(test_data["stack"], name="test_stack")
    viewer.add_labels(test_data["mask"], name="test_mask")
    estimator = MidplaneEstimator(test_data["mask"], symmetry_axis="x")
    points = estimator.get_points()

    if rotation_needed:
        points[0] += [1, 3, 2]

    viewer.add_points(points, name="test_points-midplane")
    align_widget = AlignMidplane(viewer)
    viewer.window.add_dock_widget(align_widget)

    try:
        align_widget._on_align_button_click()
    except Exception as e:
        pytest.fail(f'_on_align_button_click raised "{e}"')

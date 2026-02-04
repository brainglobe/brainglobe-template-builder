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


def test_estimate_points_different_axes(align_widget):
    """Test that different axes produce different midplane point estimates."""
    points = {}

    for axis in ["x", "y", "z"]:
        align_widget.select_axis_dropdown.setCurrentText(axis)
        align_widget._on_estimate_button_click()
        points_layer = align_widget.viewer.layers[-1]
        points[axis] = points_layer.data.copy()

    # All axis pairs should produce different points
    x, y, z = points["x"], points["y"], points["z"]
    assert not (
        np.array_equal(x, y) or np.array_equal(x, z) or np.array_equal(y, z)
    )


@pytest.mark.xfail(
    reason="bug: test transform does not result in valid transformation matrix"
)
@pytest.mark.parametrize(
    "stack_type",
    [
        pytest.param("", id="parallel midplane"),
        pytest.param("_rotated", id="rotated midplane"),
    ],
)
def test_align_midplane(make_napari_viewer, test_data, stack_type):
    """Test that align midplane modifies image and mask data correctly."""
    viewer = make_napari_viewer()
    viewer.add_image(test_data["stack" + stack_type], name="test_stack")
    viewer.add_labels(test_data["mask" + stack_type], name="test_mask")

    # Add midplane points (required for alignment)
    mask_data = test_data["mask" + stack_type]
    estimator = MidplaneEstimator(mask_data, symmetry_axis="x")
    points = estimator.get_points()
    viewer.add_points(points, name="test_points-midplane")

    align_widget = AlignMidplane(viewer)
    viewer.window.add_dock_widget(align_widget)

    try:
        align_widget._on_align_button_click()
    except Exception as e:
        pytest.fail(f'_on_align_button_click raised "{e}"')

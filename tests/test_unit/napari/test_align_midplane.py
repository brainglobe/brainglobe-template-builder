import numpy as np
import pytest

from brainglobe_template_builder.napari.align_widget import AlignMidplane
from brainglobe_template_builder.utils.alignment import (
    MidplaneEstimator,
)


@pytest.fixture
def test_data():
    """Create asymmetric test data with stack and mask."""
    stack = np.zeros((50, 50, 50), dtype=np.float32)
    stack[10:30, 15:40, 20:45] = 1.0
    mask = (stack > 0).astype(np.uint8)
    return {"stack": stack, "mask": mask}


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


@pytest.mark.xfail(reason="invalid edge_width property in widget points layer")
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


@pytest.mark.xfail(reason="invalid edge_width property in widget points layer")
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

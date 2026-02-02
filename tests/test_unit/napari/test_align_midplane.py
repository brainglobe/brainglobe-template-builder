import numpy as np
import pytest

from brainglobe_template_builder.napari.align_widget import AlignMidplane
from brainglobe_template_builder.utils.alignment import (
    MidplaneEstimator,
)


@pytest.fixture
def test_data():
    """Create test data with stack and mask."""
    stack = np.zeros((50, 50, 50), dtype=np.float32)
    stack[15:35, 15:35, 15:35] = 1.0
    mask = (stack > 0).astype(np.uint8)
    points = MidplaneEstimator(mask, symmetry_axis="x").get_points()
    return {"stack": stack, "mask": mask, "points": points}


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


# mark as xfail
@pytest.mark.xfail(reason="invalid edge_width property in widget points layer")
def test_estimate_points(align_widget, test_data):
    """Test that estimate points creates a Points layer with correct data."""
    align_widget._on_estimate_button_click()
    points_layer = align_widget.viewer.layers[-1]  # Last added layer

    # get default (x-axis) estimated points
    midplane_estimate = MidplaneEstimator(test_data["mask"], symmetry_axis="x")
    expected_points = midplane_estimate.get_points()

    assert points_layer is not None
    np.testing.assert_array_equal(points_layer.data, expected_points)

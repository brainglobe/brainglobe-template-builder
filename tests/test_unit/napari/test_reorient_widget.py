import numpy as np
import pytest
from brainglobe_space import AnatomicalSpace

from brainglobe_template_builder.napari.reorient_widget import Reorient


@pytest.fixture
def reorient_widget(make_napari_viewer):
    """
    Create a viewer, add the re-orient widget, and return the widget.
    The viewer can be accessed using ``widget.viewer``.
    """
    viewer = make_napari_viewer()
    widget = Reorient(viewer)

    # Starting orientation = SAL
    widget.source_origin1.setCurrentText("s")
    widget.source_origin2.setCurrentText("a")
    widget.source_origin3.setCurrentText("l")

    # End orientation = ASR
    widget.target_origin1.setCurrentText("a")
    widget.target_origin2.setCurrentText("s")
    widget.target_origin3.setCurrentText("r")

    viewer.window.add_dock_widget(widget)

    return widget


@pytest.fixture()
def stack() -> np.ndarray:
    """Create 50x50x50 stack with a small off-centre object (value 0.5)."""
    stack = np.zeros((50, 50, 50))
    stack[10:30, 10:30, 10:30] = 0.5
    return stack


@pytest.fixture()
def points() -> np.ndarray:
    """Two example 3D points."""

    return np.array(
        [
            [2, 3, 4],
            [5, 6, 7],
        ]
    )


def test_reorient_stack(reorient_widget, stack):
    """Test the widget correctly re-orients ONLY the selected
    image from SAL -> ASR"""
    viewer = reorient_widget.viewer

    # Add two images - we will re-orient one of them
    viewer.add_image(stack, name="stack_to_reorient")
    viewer.add_image(stack, name="stack_to_NOT_reorient")

    # Select only one of the layers
    viewer.layers.selection.select_only(viewer.layers["stack_to_reorient"])

    reorient_widget.reorient_layers()

    reoriented_image = viewer.layers["stack_to_reorient_orig-asr"].data
    not_reoriented_image = viewer.layers["stack_to_NOT_reorient"].data
    expected_reoriented_image = AnatomicalSpace("SAL").map_stack_to(
        "ASR", stack
    )

    assert len(viewer.layers) == 2
    np.testing.assert_array_equal(reoriented_image, expected_reoriented_image)
    np.testing.assert_array_equal(not_reoriented_image, stack)


def test_reorient_only_points(reorient_widget, points):
    """Test that no re-orientation occurs when only a points layer
    is selected."""
    viewer = reorient_widget.viewer

    # Add points layer
    layer_name = "points_to_reorient"
    points_layer = viewer.add_points(points, name=layer_name)
    viewer.layers.selection.select_only(points_layer)

    reorient_widget.reorient_layers()

    # No re-orientation should occur when only a points
    # layer is selected
    assert len(viewer.layers) == 1
    np.testing.assert_array_equal(viewer.layers[layer_name].data, points)


def test_reorient_stack_and_points(reorient_widget, points, stack):
    """Test that points are correctly re-oriented when both a points + image
    layer are selected."""
    viewer = reorient_widget.viewer

    # Add points and stack
    viewer.add_image(stack, name="stack_to_reorient")
    viewer.add_points(points, name="points_to_reorient")

    viewer.layers.select_all()
    reorient_widget.reorient_layers()

    # Points should have been re-oriented to match stack
    expected_points = np.array([[3.0, 2.0, 46.0], [6.0, 5.0, 43.0]])
    assert len(viewer.layers) == 2
    np.testing.assert_array_equal(
        viewer.layers["points_to_reorient_space-asr"].data, expected_points
    )

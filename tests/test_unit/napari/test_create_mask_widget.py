import pytest

from brainglobe_template_builder.napari.mask_widget import CreateMask


@pytest.fixture
def mask_widget(make_napari_viewer, stack):
    """Creates a napari viewer with the CreateMask widget
    docked and a test stack layer added."""
    viewer = make_napari_viewer()
    viewer.add_image(stack, name="test_stack")
    mask_widget = CreateMask(viewer)
    viewer.window.add_dock_widget(mask_widget)
    return mask_widget


def test_create_mask_creates_layer(mask_widget):
    """Test that clicking 'Create mask' generates a new layer."""
    initial_layer_count = len(mask_widget.viewer.layers)
    mask_widget._on_create_mask_button_click()
    assert len(mask_widget.viewer.layers) == initial_layer_count + 1

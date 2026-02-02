import numpy as np
import pytest

from brainglobe_template_builder.napari.mask_widget import CreateMask
from brainglobe_template_builder.utils.masking import create_mask


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


@pytest.mark.parametrize(
    "mask_config",
    [
        pytest.param(
            {
                "default": {
                    "gauss_sigma": 3,
                    "threshold_method": "triangle",
                    "closing_size": 5,
                    "erode_size": 0,
                }
            },
            id="default",
        ),
        pytest.param(
            {
                "custom": {
                    "gauss_sigma": 1,
                    "threshold_method": "otsu",
                    "closing_size": 3,
                    "erode_size": 2,
                }
            },
            id="custom otsu",
        ),
    ],
)
def test_create_mask_layer_data(mask_widget, stack, mask_config):
    """Test that the created mask layer contains the correct data.

    Applying different masking configurations (default and custom)
    produce a mask layer with data matching the expected output.
    """

    if "default" in mask_config:
        mask_config = mask_config["default"]

    elif "custom" in mask_config:
        mask_config = mask_config["custom"]
        mask_widget.gauss_sigma.setValue(mask_config["gauss_sigma"])
        mask_widget.threshold_method.setCurrentText(
            mask_config["threshold_method"]
        )
        mask_widget.closing_size.setValue(mask_config["closing_size"])
        mask_widget.erode_size.setValue(mask_config["erode_size"])

    expected_mask_data = create_mask(stack, **mask_config)

    mask_widget._on_create_mask_button_click()
    mask_layer = mask_widget.viewer.layers[-1]

    np.testing.assert_array_equal(mask_layer.data, expected_mask_data)


@pytest.mark.parametrize(
    "label,input_val,expected_val",
    [
        pytest.param("gauss_sigma", -5, 0, id="gauss_sigma min"),
        pytest.param("gauss_sigma", 25, 20, id="gauss_sigma max"),
        pytest.param("closing_size", -5, 0, id="closing_size min"),
        pytest.param("closing_size", 25, 20, id="closing_size max"),
        pytest.param("erode_size", -5, 0, id="erode_size min"),
        pytest.param("erode_size", 25, 20, id="erode_size max"),
    ],
)
def test_create_mask_value_clamp(mask_widget, label, input_val, expected_val):
    """Test that clamping of valid values works correctly."""
    spinbox = getattr(mask_widget, label)
    spinbox.setValue(input_val)
    assert spinbox.value() == expected_val

import numpy as np
import pytest

from brainglobe_template_builder.napari.brightness_correction_widget import (
    CorrectBrightness,
)
from brainglobe_template_builder.utils.brightness import (
    correct_image_brightness,
)


@pytest.fixture
def brightness_correction_widget(make_napari_viewer, stack):
    """Creates a napari viewer with the BrightnessCorrection widget
    docked and a test stack layer added."""
    viewer = make_napari_viewer()
    viewer.add_image(stack, name="test_stack")
    correct_brightness_widget = CorrectBrightness(viewer)
    viewer.window.add_dock_widget(correct_brightness_widget)
    return correct_brightness_widget


def test_brightness_correction_creates_layer(brightness_correction_widget):
    """Test that clicking 'Correct brightness' generates a new layer."""
    initial_layer_count = len(brightness_correction_widget.viewer.layers)
    brightness_correction_widget._on_correct_brightness_button_click()
    assert (
        len(brightness_correction_widget.viewer.layers)
        == initial_layer_count + 1
    )


def test_correct_brightness_layer_data_default(
    brightness_correction_widget, stack
):
    """Test that the corrected brightness layer contains the correct data
    with default configuration."""
    expected_corrected_data = correct_image_brightness(stack, [1, 1, 1])
    brightness_correction_widget._on_correct_brightness_button_click()
    corrected_layer = brightness_correction_widget.viewer.layers[-1]

    np.testing.assert_array_equal(
        corrected_layer.data, expected_corrected_data
    )


def test_correct_brightness_layer_data_anisotropic(
    brightness_correction_widget, stack
):
    """Test that the corrected brightness layer with anisotropic voxels
    matches the underlying function call."""

    expected_corrected_anisotropic_data = correct_image_brightness(
        stack, [0.04, 0.03, 0.01]
    )

    brightness_correction_widget.voxel_size_widget.set_voxel_sizes(
        [0.04, 0.03, 0.01]
    )
    brightness_correction_widget._on_correct_brightness_button_click()
    corrected_layer_anisotropic = brightness_correction_widget.viewer.layers[
        -1
    ]

    np.testing.assert_array_equal(
        corrected_layer_anisotropic.data, expected_corrected_anisotropic_data
    )


def test_correct_brightness_layer_data_anisotropic_is_not_default(
    brightness_correction_widget, stack
):
    """Test that the corrected brightness layer with anisotropic voxels
    is different from the default"""
    expected_corrected_anisotropic_data = correct_image_brightness(
        stack, [0.04, 0.03, 0.01]
    )

    brightness_correction_widget._on_correct_brightness_button_click()
    corrected_layer_default = brightness_correction_widget.viewer.layers[-1]
    assert not np.array_equal(
        corrected_layer_default.data, expected_corrected_anisotropic_data
    )

import numpy as np
import pytest
import yaml

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


@pytest.fixture
def default_mask_config_kwargs():
    """Returns the default mask configuration as a dictionary."""
    return {
        "closing_size": 5,
        "erode_size": 0,
        "gauss_sigma": 3.0,
        "threshold_method": "triangle",
    }


def test_create_mask_creates_layer(mask_widget):
    """Test that clicking 'Create mask' generates a new layer."""
    initial_layer_count = len(mask_widget.viewer.layers)
    mask_widget._on_create_mask_button_click()
    assert len(mask_widget.viewer.layers) == initial_layer_count + 1


def test_create_mask_layer_data_default(
    mask_widget, stack, default_mask_config_kwargs
):
    """Test that the created mask layer contains the correct data
    with default configuration."""
    expected_mask_data = create_mask(stack, **default_mask_config_kwargs)
    mask_widget._on_create_mask_button_click()
    mask_layer = mask_widget.viewer.layers[-1]

    np.testing.assert_array_equal(mask_layer.data, expected_mask_data)


@pytest.mark.parametrize(
    "mask_config_kwargs",
    [
        pytest.param(
            {
                "gauss_sigma": 1,
                "threshold_method": "otsu",
                "closing_size": 3,
                "erode_size": 2,
            },
            id="custom otsu",
        ),
        pytest.param(
            {
                "gauss_sigma": 2,
                "threshold_method": "isodata",
                "closing_size": 4,
                "erode_size": 2,
            },
            id="custom isodata",
        ),
    ],
)
def test_create_mask_layer_data_custom(mask_widget, stack, mask_config_kwargs):
    """Test that the created mask layer contains the correct data
    with various custom configurations."""
    mask_widget.gauss_sigma.setValue(mask_config_kwargs["gauss_sigma"])
    mask_widget.threshold_method.setCurrentText(
        mask_config_kwargs["threshold_method"]
    )
    mask_widget.closing_size.setValue(mask_config_kwargs["closing_size"])
    mask_widget.erode_size.setValue(mask_config_kwargs["erode_size"])

    expected_mask_data = create_mask(stack, **mask_config_kwargs)

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


@pytest.mark.parametrize(
    "pad",
    [
        pytest.param(5, id="pad_pixels 5"),
        pytest.param(10, id="pad_pixels 10"),
    ],
)
def test_export_mask_config(
    mask_widget, tmp_path, default_mask_config_kwargs, pad
):
    """Test that exporting mask configuration to a yaml file
    works as expected."""
    output_path = tmp_path / "output"
    output_path.mkdir(exist_ok=True)

    mask_widget.output_dir_widget.path_edit.setText(str(output_path))
    mask_widget.config_dir_widget.path_edit.setText(str(tmp_path))
    mask_widget.pad_pixels.setValue(pad)
    mask_widget._on_export_config_button_click()
    config_file = tmp_path / "preproc_config.yaml"

    # change keyname gauss_sigma (internal use) to gaussian_sigma (config file)
    mask_config = default_mask_config_kwargs.copy()
    mask_config["gaussian_sigma"] = mask_config.pop("gauss_sigma")

    expected_config = {
        "mask": mask_config,
        "output_dir": str(output_path),
        "pad_pixels": pad,
    }

    assert config_file.exists(), "Config file was not created."
    with open(config_file, "r") as f:
        exported_config = yaml.safe_load(f)
    assert exported_config == expected_config, "Exported config is incorrect."

"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari

from napari import Viewer
from napari.layers import Image
from napari.types import LayerDataTuple


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory(
    call_button="gaussian blur",
    sigma={"widget_type": "FloatSlider", "max": 10, "min": 0, "step": 1},
    tooltips={"sigma": "Standard deviation for Gaussian kernel (in pixels)"},
    # do not show the viewer argument, as it is not a magicgui argument
)
def gaussian_blur_widget(
    viewer: Viewer, image: Image, sigma: float = 3
) -> LayerDataTuple:
    """Smooth image with a gaussian filter and add to napari viewer.

    Parameters
    ----------
    image : Image
        A napari image layer to smooth.
    sigma : float
        Standard deviation for Gaussian kernel (in pixels).

    Returns
    -------
    image_smoothed : Image
        A napari image layer with the smoothed image.
    """

    if image is not None:
        assert isinstance(image, Image), "image must be a napari Image layer"
    else:
        print("Please select an image layer")
        return

    from skimage import filters

    image_smoothed = filters.gaussian(image.data, sigma=sigma)

    # return the smoothed image layer
    return (
        image_smoothed,
        {"name": f"{image.name}_smoothed"},
        "image",
    )


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return gaussian_blur_widget

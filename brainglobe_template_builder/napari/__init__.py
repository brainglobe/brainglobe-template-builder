import setuptools_scm

try:
    release = setuptools_scm.get_version(root="../..", relative_to=__file__)
    __version__ = release.split("+")[0]  # remove git hash
except LookupError:
    __version__ = "unknown"

from brainglobe_template_builder.napari._reader import napari_get_reader
from brainglobe_template_builder.napari._widget import mask_widget

__all__ = (
    "napari_get_reader",
    "mask_widget",
)

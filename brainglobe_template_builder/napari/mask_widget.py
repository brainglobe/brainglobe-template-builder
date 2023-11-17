from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtWidgets import QComboBox, QFormLayout, QWidget


class GenerateMask(QWidget):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)

        self.setLayout(QFormLayout())

        image_layer_names = [
            layer.name
            for layer in napari_viewer.layers
            if isinstance(layer, Image)
        ]
        self.image_selector = QComboBox(parent=self)
        self.image_selector.addItems(image_layer_names)
        self.layout().addRow("image:", self.image_selector)

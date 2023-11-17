import numpy as np
from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QWidget,
)
from skimage import filters, morphology

from brainglobe_template_builder.utils import (
    extract_largest_object,
    threshold_image,
)


class GenerateMask(QWidget):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        self.gauss_sigma = QSpinBox(parent=self)
        self.gauss_sigma.setRange(0, 20)
        self.gauss_sigma.setValue(3)
        self.layout().addRow("gauss sigma:", self.gauss_sigma)

        self.threshold_method = QComboBox(parent=self)
        self.threshold_method.addItems(["triangle", "otsu", "isodata"])
        self.layout().addRow("threshold method:", self.threshold_method)

        self.erosion_size = QSpinBox(parent=self)
        self.erosion_size.setRange(0, 20)
        self.erosion_size.setValue(5)
        self.layout().addRow("erosion size:", self.erosion_size)

        self.generate_mask_button = QPushButton("Generate mask", parent=self)
        self.layout().addRow(self.generate_mask_button)
        self.generate_mask_button.clicked.connect(self._on_button_click)

    def _on_button_click(self):
        """Generate a mask from the selected image layer."""

        if len(self.viewer.layers.selection) != 1:
            print("Please select exactly one image layer")
            return None
        else:
            image = list(self.viewer.layers.selection)[0]

        if not isinstance(image, Image):
            print("The selected layer is not an image layer")
            return None

        # Get parameters from widgets
        gauss_sigma = self.gauss_sigma.value()
        threshold_method = self.threshold_method.currentText()
        erosion_size = self.erosion_size.value()

        # Apply gaussian filter to image
        if gauss_sigma > 0:
            data_smoothed = filters.gaussian(image.data, sigma=gauss_sigma)
        else:
            data_smoothed = image.data

        # Threshold the (smoothed) image
        binary = threshold_image(data_smoothed, method=threshold_method)

        # Keep only the largest object in the binary image
        mask = extract_largest_object(binary)

        # Erode the mask
        if erosion_size > 0:
            mask = morphology.binary_erosion(
                mask, footprint=np.ones((erosion_size,) * image.ndim)
            )

        self.viewer.add_labels(mask, opacity=0.5, name="mask")

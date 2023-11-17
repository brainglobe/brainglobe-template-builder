import numpy as np
from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidget
from napari.layers import Image
from napari.utils.notifications import show_info
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


class GenerateMask(CollapsibleWidget):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(title="Generate Mask", parent=parent)
        self.viewer = napari_viewer

        content = QWidget(parent=self)
        content.setLayout(QFormLayout())

        self.setContent(content)

        self.gauss_sigma = QSpinBox(parent=self)
        self.gauss_sigma.setRange(0, 20)
        self.gauss_sigma.setValue(3)
        content.layout().addRow("gauss sigma:", self.gauss_sigma)

        self.threshold_method = QComboBox(parent=self)
        self.threshold_method.addItems(["triangle", "otsu", "isodata"])
        content.layout().addRow("threshold method:", self.threshold_method)

        self.erosion_size = QSpinBox(parent=self)
        self.erosion_size.setRange(0, 20)
        self.erosion_size.setValue(5)
        content.layout().addRow("erosion size:", self.erosion_size)

        self.generate_mask_button = QPushButton("Generate mask", parent=self)
        content.layout().addRow(self.generate_mask_button)
        self.generate_mask_button.clicked.connect(self._on_button_click)

    def _on_button_click(self):
        """Generate a mask from the selected image layer."""

        if len(self.viewer.layers.selection) != 1:
            show_info("Please select exactly one image layer")
            return None

        image = list(self.viewer.layers.selection)[0]

        if not isinstance(image, Image):
            show_info("The selected layer is not an image layer")
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

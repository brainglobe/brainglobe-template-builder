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

from brainglobe_template_builder.preproc import create_mask


class CreateMask(QWidget):
    """Widget to create a mask from a selected image layer."""

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

        self.generate_mask_button = QPushButton("Create mask", parent=self)
        self.layout().addRow(self.generate_mask_button)
        self.generate_mask_button.clicked.connect(self._on_button_click)

    def _on_button_click(self):
        """Create a mask from the selected image layer, using the parameters
        specified in the widget, and add it to the napari viewer.
        """

        if len(self.viewer.layers.selection) != 1:
            show_info("Please select exactly one Image layer")
            return None

        image = list(self.viewer.layers.selection)[0]

        if not isinstance(image, Image):
            show_info("The selected layer is not an Image layer")
            return None

        mask_data = create_mask(
            image.data,
            gauss_sigma=self.gauss_sigma.value(),
            threshold_method=self.threshold_method.currentText(),
            erosion_size=self.erosion_size.value(),
        )

        self.viewer.add_labels(mask_data, name="mask", opacity=0.5)

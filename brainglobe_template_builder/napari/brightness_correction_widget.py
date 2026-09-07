from napari.layers import Image
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from brainglobe_template_builder.utils.brightness import (
    correct_image_brightness,
)


class CorrectBrightness(QWidget):
    """Widget to apply N4 bias field brightness correction
    to a selected image layer."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        self._correct_brightness_group()

    def _correct_brightness_group(self):
        """Create the group of widgets concerned with correcting brightness."""
        self.correct_brightness_groupbox = QGroupBox("Correct brightness")
        self.correct_brightness_groupbox.setLayout(QFormLayout())
        self.layout().addRow(self.correct_brightness_groupbox)

        self.voxel_size = QSpinBox(parent=self.correct_brightness_groupbox)
        self.voxel_size.setRange(0, 20)
        self.voxel_size.setValue(1)
        self.correct_brightness_groupbox.layout().addRow(
            "Voxel size (microns):", self.voxel_size
        )

        self.correct_brightness_button = QPushButton(
            "Correct Brightness", parent=self
        )
        self.correct_brightness_groupbox.layout().addRow(
            self.correct_brightness_button
        )
        self.correct_brightness_button.clicked.connect(
            self._on_correct_brightness_button_click
        )

    def _on_correct_brightness_button_click(self):
        """Correct brightness in the selected image layer."""

        if len(self.viewer.layers.selection) != 1:
            show_info("Please select exactly one Image layer")
            return None

        image = list(self.viewer.layers.selection)[0]

        if not isinstance(image, Image):
            show_info("The selected layer is not an Image layer")
            return None

        spacing = [self.voxel_size.value()] * 3
        corrected_name = f"{image.name}_brightness-corrected"

        corrected_image_data = correct_image_brightness(
            image.data, spacing=spacing
        )

        self.viewer.add_image(
            corrected_image_data, name=corrected_name, scale=image.scale
        )

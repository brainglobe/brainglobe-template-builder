from napari.layers import Image
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QPushButton,
    QWidget,
)

from brainglobe_template_builder.napari.utils import VoxelSizeWidget
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

        self._create_voxel_size_widget()

        self.correct_brightness_button = QPushButton(
            "Correct Brightness", parent=self
        )
        self.correct_brightness_groupbox.layout().addRow(
            self.correct_brightness_button
        )
        self.correct_brightness_button.clicked.connect(
            self._on_correct_brightness_button_click
        )

    def _create_voxel_size_widget(self):
        """Create 3 fields for entering the voxel size."""
        self.voxel_size_widget = VoxelSizeWidget()
        self.correct_brightness_groupbox.layout().addRow(
            "Voxel size (axes 0, 1, 2) in mm:", self.voxel_size_widget
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

        corrected_name = f"{image.name}_brightness-corrected"

        spacing = self.voxel_size_widget.get_voxel_sizes()
        corrected_image_data = correct_image_brightness(
            image.data, spacing=spacing
        )

        self.viewer.add_image(
            corrected_image_data, name=corrected_name, scale=image.scale
        )

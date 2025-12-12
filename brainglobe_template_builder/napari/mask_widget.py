from pathlib import Path

import yaml
from napari.layers import Image
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from brainglobe_template_builder.napari.utils import DirPathWidget
from brainglobe_template_builder.preproc import create_mask
from brainglobe_template_builder.utils.preproc_config import (
    MaskConfig,
    PreprocConfig,
)


class CreateMask(QWidget):
    """Widget to create a mask from a selected image layer."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        self._create_mask_group()
        self._create_export_config_group()

    def _create_mask_group(self):
        """Create the group of widgets concerned with creating a mask."""
        self.mask_groupbox = QGroupBox("Create mask to exclude background")
        self.mask_groupbox.setLayout(QFormLayout())
        self.layout().addRow(self.mask_groupbox)

        self.gauss_sigma = QSpinBox(parent=self.mask_groupbox)
        self.gauss_sigma.setRange(0, 20)
        self.gauss_sigma.setValue(3)
        self.mask_groupbox.layout().addRow("gaussian sigma:", self.gauss_sigma)

        self.threshold_method = QComboBox(parent=self.mask_groupbox)
        self.threshold_method.addItems(["triangle", "otsu", "isodata"])
        self.mask_groupbox.layout().addRow(
            "threshold method:", self.threshold_method
        )

        self.closing_size = QSpinBox(parent=self.mask_groupbox)
        self.closing_size.setRange(0, 20)
        self.closing_size.setValue(5)
        self.mask_groupbox.layout().addRow("closing size:", self.closing_size)

        self.erode_size = QSpinBox(parent=self.mask_groupbox)
        self.erode_size.setRange(0, 20)
        self.erode_size.setValue(0)
        self.mask_groupbox.layout().addRow("erode size:", self.erode_size)

        self.create_mask_button = QPushButton("Create mask", parent=self)
        self.mask_groupbox.layout().addRow(self.create_mask_button)
        self.create_mask_button.clicked.connect(
            self._on_create_mask_button_click
        )

    def _create_export_config_group(self):
        """Create the group of widgets concerned with exporting mask settings
        to a yaml config file."""

        self.config_groupbox = QGroupBox("Export mask settings to config file")
        self.config_groupbox.setLayout(QFormLayout())
        self.layout().addRow(self.config_groupbox)

        self.pad_pixels = QSpinBox(parent=self.config_groupbox)
        self.pad_pixels.setRange(0, 100)
        self.pad_pixels.setValue(5)
        self.config_groupbox.layout().addRow("pad pixels:", self.pad_pixels)

        self.output_dir_widget = DirPathWidget(
            self.config_groupbox, "Preprocess output directory:"
        )
        self.config_dir_widget = DirPathWidget(
            self.config_groupbox, "Config file directory:"
        )

        self.export_config_button = QPushButton("Export config", parent=self)
        self.config_groupbox.layout().addRow(self.export_config_button)
        self.export_config_button.clicked.connect(
            self._on_export_config_button_click
        )

    def _on_create_mask_button_click(self):
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
            closing_size=self.closing_size.value(),
            erode_size=self.erode_size.value(),
        )

        mask_name = f"{image.name}_label-brain"
        self.viewer.add_labels(
            mask_data, name=mask_name, opacity=0.5, scale=image.scale
        )

    def _on_export_config_button_click(self):
        """Export a yaml config file, using the mask parameters
        specified in the widget.
        """

        output_dir = self.output_dir_widget.get_dir_path()
        config_dir = self.config_dir_widget.get_dir_path()

        if not config_dir or not output_dir:
            return

        config = PreprocConfig(
            mask=MaskConfig(
                gaussian_sigma=self.gauss_sigma.value(),
                threshold_method=self.threshold_method.currentText(),
                closing_size=self.closing_size.value(),
                erode_size=self.erode_size.value(),
            ),
            output_dir=output_dir,
            pad_pixels=self.pad_pixels.value(),
        )

        config_path = Path(config_dir) / "preproc_config.yaml"
        with open(config_path, "w") as outfile:
            yaml.dump(config.model_dump(mode="json"), outfile)

        info_msg = "Created preproc_config.yaml."
        show_info(info_msg)

from brainglobe_space import AnatomicalSpace
from napari.layers import Image, Labels, Points
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)


class Reorient(QWidget):
    """Widget to reorient images to a standard anatomical space."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        self._create_reorient_group()

    def _create_reorient_group(self):
        self.reorient_groupbox = QGroupBox("Reorient to standard space")
        self.reorient_groupbox.setLayout(QFormLayout())
        self.layout().addRow(self.reorient_groupbox)

        avail_labels = AnatomicalSpace.lims_labels

        # Enter labels for source space
        source_origin_row = QHBoxLayout()
        self.source_origin1 = QComboBox(parent=self.reorient_groupbox)
        self.source_origin1.addItems(avail_labels)
        self.source_origin1.setCurrentIndex(0)
        source_origin_row.addWidget(self.source_origin1)
        self.source_origin2 = QComboBox(parent=self.reorient_groupbox)
        self.source_origin2.addItems(avail_labels)
        self.source_origin2.setCurrentIndex(2)
        source_origin_row.addWidget(self.source_origin2)
        self.source_origin3 = QComboBox(parent=self.reorient_groupbox)
        self.source_origin3.addItems(avail_labels)
        self.source_origin3.setCurrentIndex(5)
        source_origin_row.addWidget(self.source_origin3)
        self.reorient_groupbox.layout().addRow(
            "Source origin:", source_origin_row
        )

        # Enter labels for target space
        target_origin_row = QHBoxLayout()
        self.target_origin1 = QComboBox(parent=self.reorient_groupbox)
        self.target_origin1.addItems(avail_labels)
        self.target_origin1.setCurrentIndex(1)
        target_origin_row.addWidget(self.target_origin1)
        self.target_origin2 = QComboBox(parent=self.reorient_groupbox)
        self.target_origin2.addItems(avail_labels)
        self.target_origin2.setCurrentIndex(2)
        target_origin_row.addWidget(self.target_origin2)
        self.target_origin3 = QComboBox(parent=self.reorient_groupbox)
        self.target_origin3.addItems(avail_labels)
        self.target_origin3.setCurrentIndex(5)
        target_origin_row.addWidget(self.target_origin3)
        self.reorient_groupbox.layout().addRow(
            "Target origin:", target_origin_row
        )

        self.reorient_button = QPushButton(
            "Reorient selected layers", parent=self.reorient_groupbox
        )
        self.reorient_button.clicked.connect(self.reorient_layers)
        self.reorient_groupbox.layout().addRow(self.reorient_button)

    def reorient_layers(self):
        """Reorient selected layers to the target space."""
        selected_layers = self.viewer.layers.selection
        selected_stacks = [
            layer
            for layer in selected_layers
            if isinstance(layer, Image) or isinstance(layer, Labels)
        ]
        selected_points = [
            layer for layer in selected_layers if isinstance(layer, Points)
        ]

        if not selected_layers:
            show_info("No layers selected. Select layers to reorient.")
            return

        source_origin = [
            self.source_origin1.currentText(),
            self.source_origin2.currentText(),
            self.source_origin3.currentText(),
        ]
        target_origin = [
            self.target_origin1.currentText(),
            self.target_origin2.currentText(),
            self.target_origin3.currentText(),
        ]
        source_origin = "".join(source_origin)
        target_origin = "".join(target_origin)

        for layer in selected_points:
            if not selected_stacks:
                show_info(
                    "Cannot reorient Points layer alone. Please select "
                    "at least one Image or Labels layer along with the "
                    "Points layer."
                )
                return
            source_stack_shape = selected_stacks[0].data.shape
            source_space = AnatomicalSpace(
                source_origin, shape=source_stack_shape
            )
            layer.data = source_space.map_points_to(target_origin, layer.data)
            layer.name = f"{layer.name}_space-{target_origin}"

        for layer in selected_stacks:
            interp_order = 0 if isinstance(layer, Labels) else 3
            source_space = AnatomicalSpace(
                source_origin, shape=layer.data.shape
            )
            layer.data = source_space.map_stack_to(
                target_origin, layer.data, interp_order=interp_order
            )
            layer.name = f"{layer.name}_orig-{target_origin}"

        show_info("Layers reoriented to target space.")

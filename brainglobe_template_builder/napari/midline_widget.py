from napari.layers import Image, Labels, Layer, Points
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QWidget,
)

from brainglobe_template_builder.preproc import (
    apply_transform,
    get_alignment_transform,
    get_midline_points,
)


class FindMidline(QWidget):
    """Widget to find the mid-sagittal plane based on annotated points."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        self._create_estimate_group()
        self._create_align_group()

    def _create_estimate_group(self):
        """Create the group of widgets concerned with estimating midline
        points."""
        self.estimate_groupbox = QGroupBox("Estimate points along midline")
        self.estimate_groupbox.setLayout(QFormLayout())
        self.layout().addRow(self.estimate_groupbox)

        # Add dropdown to select labels layer (mask)
        self.select_mask_dropdown = QComboBox(parent=self.estimate_groupbox)
        self.select_mask_dropdown.addItems(self._get_layers_by_type(Labels))
        self.select_mask_dropdown.currentTextChanged.connect(
            self._on_dropdown_selection_change
        )
        self.estimate_groupbox.layout().addRow(
            "mask:", self.select_mask_dropdown
        )

        # Initialise button to estimate midline points
        self.estimate_points_button = QPushButton(
            "Estimate points", parent=self.estimate_groupbox
        )
        self.estimate_points_button.setEnabled(False)
        self.estimate_points_button.clicked.connect(
            self._on_estimate_button_click
        )
        self.estimate_groupbox.layout().addRow(self.estimate_points_button)

    def _create_align_group(self):
        """Create the group of widgets concerned with aligning the image to
        the midline."""

        self.align_groupbox = QGroupBox("Align image to midline")
        self.align_groupbox.setLayout(QFormLayout())
        self.layout().addRow(self.align_groupbox)

        # Add dropdown to select image layer
        self.select_image_dropdown = QComboBox(parent=self.align_groupbox)
        self.select_image_dropdown.addItems(self._get_layers_by_type(Image))
        self.select_image_dropdown.currentTextChanged.connect(
            self._on_dropdown_selection_change
        )
        self.align_groupbox.layout().addRow(
            "image:", self.select_image_dropdown
        )

        # Add dropdown to select points layer
        self.select_points_dropdown = QComboBox(parent=self.align_groupbox)
        self.select_points_dropdown.addItems(self._get_layers_by_type(Points))
        self.select_points_dropdown.currentTextChanged.connect(
            self._on_dropdown_selection_change
        )
        self.align_groupbox.layout().addRow(
            "points:", self.select_points_dropdown
        )

        # Add dropdown to select axis
        self.select_axis_dropdown = QComboBox(parent=self.align_groupbox)
        self.select_axis_dropdown.addItems(["x", "y", "z"])
        self.align_groupbox.layout().addRow("axis:", self.select_axis_dropdown)

        # Add button to align image to midline
        self.align_image_button = QPushButton(
            "Align image", parent=self.align_groupbox
        )
        self.align_image_button.setEnabled(False)
        self.align_image_button.clicked.connect(self._on_align_button_click)
        self.align_groupbox.layout().addRow(self.align_image_button)

        # 9 colors taken from ColorBrewer2.org Set3 palette
        self.point_colors = [
            "#8dd3c7",
            "#ffffb3",
            "#bebada",
            "#fb8072",
            "#80b1d3",
            "#fdb462",
            "#b3de69",
            "#fccde5",
            "#d9d9d9",
        ]

    def _get_layers_by_type(self, layer_type: Layer) -> list:
        """Return a list of napari layers of a given type."""
        return [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, layer_type)
        ]

    def refresh_dropdowns(self):
        """Refresh the dropdowns to reflect the current layers."""
        for layer_type, dropdown in zip(
            [Labels, Image, Points],
            [
                self.select_mask_dropdown,
                self.select_image_dropdown,
                self.select_points_dropdown,
            ],
        ):
            dropdown.clear()
            dropdown.addItems(self._get_layers_by_type(layer_type))

    def _on_estimate_button_click(self):
        """Estimate midline points and add them to the viewer."""

        # Estimate 9 midline points based on the selected mask
        mask_name = self.select_mask_dropdown.currentText()
        mask = self.viewer.layers[mask_name]
        points = get_midline_points(mask.data)
        # Point layer attributes
        point_attrs = {
            "properties": {"label": range(1, points.shape[0] + 1)},
            "face_color": "label",
            "face_color_cycle": self.point_colors,
            "symbol": "cross",
            "edge_width": 0,
            "opacity": 0.6,
            "size": 6,
            "ndim": mask.ndim,
            "name": "midline points",
        }

        mask.visible = False
        self.viewer.add_points(points, **point_attrs)
        self.refresh_dropdowns()
        show_info(
            "Please move the estimated points so that they sit exactly "
            "on the mid-sagittal plane."
        )

    def _on_align_button_click(self):
        """Align image and add the transformed image to the viewer."""
        image_name = self.select_image_dropdown.currentText()
        points_name = self.select_points_dropdown.currentText()
        axis = self.select_axis_dropdown.currentText()

        transform = get_alignment_transform(
            self.viewer.layers[image_name].data,
            self.viewer.layers[points_name].data,
            axis=axis,
        )

        aligned_image = apply_transform(
            self.viewer.layers[image_name].data, transform
        )

        self.viewer.add_image(aligned_image, name="aligned image")

    def _on_dropdown_selection_change(self):
        # Enable estimate button if mask dropdown has a value
        if self.select_mask_dropdown.currentText() == "":
            self.estimate_points_button.setEnabled(False)
        else:
            self.estimate_points_button.setEnabled(True)

        # Enable align button if both image and points dropdowns have a value
        if (
            self.select_image_dropdown.currentText() == ""
            or self.select_points_dropdown.currentText() == ""
        ):
            self.align_image_button.setEnabled(False)
        else:
            self.align_image_button.setEnabled(True)

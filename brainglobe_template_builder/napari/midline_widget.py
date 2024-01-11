from napari.layers import Image, Labels, Layer, Points
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QPushButton,
    QWidget,
)

from brainglobe_template_builder.preproc import (
    align_to_midline,
    get_midline_points,
)


class FindMidline(QWidget):
    """Widget to find the mid-sagittal plane based on annotated points."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        # Initialise button to estimate midline points
        self.estimate_points_button = QPushButton(
            "Estimate midline points", parent=self
        )
        self.layout().addRow(self.estimate_points_button)
        self.estimate_points_button.clicked.connect(
            self._on_estimate_button_click
        )

        # Add dropdown to select image layer
        self.select_image_dropdown = QComboBox(parent=self)
        self.select_image_dropdown.addItems(self._get_layers_by_type(Image))
        self.select_image_dropdown.currentTextChanged.connect(
            self._on_dropdown_selection_change
        )
        self.layout().addRow("image:", self.select_image_dropdown)

        # Add dropdown to select points layer
        self.select_points_dropdown = QComboBox(parent=self)
        self.select_points_dropdown.addItems(self._get_layers_by_type(Points))
        self.select_points_dropdown.currentTextChanged.connect(
            self._on_dropdown_selection_change
        )
        self.layout().addRow("points:", self.select_points_dropdown)

        # Add dropdown to select axis
        self.select_axis_dropdown = QComboBox(parent=self)
        self.select_axis_dropdown.addItems(["x", "y", "z"])
        self.layout().addRow("axis:", self.select_axis_dropdown)

        # Add button to align image to midline
        self.align_image_button = QPushButton(
            "Align image to midline", parent=self
        )
        self.layout().addRow(self.align_image_button)
        self.align_image_button.clicked.connect(self._on_align_button_click)
        self.align_image_button.setEnabled(False)

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

    def _refresh_layer_dropdowns(self):
        """Refresh the dropdowns to reflect the current layers."""
        for layer_type, dropdown in zip(
            [Image, Points],
            [self.select_image_dropdown, self.select_points_dropdown],
        ):
            dropdown.clear()
            dropdown.addItems(self._get_layers_by_type(layer_type))

    def _on_estimate_button_click(self):
        """Estimate midline points and add them to the viewer."""
        if len(self.viewer.layers.selection) != 1:
            show_info("Please select exactly one Labels layer")
            return None

        mask = list(self.viewer.layers.selection)[0]

        if not isinstance(mask, Labels):
            show_info("The selected layer is not a Labels layer")
            return None

        # Estimate 9 midline points
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
        self._refresh_layer_dropdowns()

    def _on_align_button_click(self):
        """Align image and add the transformed image to the viewer."""
        # Get values from dropdowns
        image_name = self.select_image_dropdown.currentText()
        points_name = self.select_points_dropdown.currentText()
        axis = self.select_axis_dropdown.currentText()

        # Call align_to_midline function
        aligned_image = align_to_midline(
            self.viewer.layers[image_name].data,
            self.viewer.layers[points_name].data,
            axis=axis,
        )

        self.viewer.add_image(aligned_image, name="aligned image")

    def _on_dropdown_selection_change(self):
        """Enable align button if both image and points dropdowns
        have a selection."""
        if (
            self.select_image_dropdown.currentText() == ""
            or self.select_points_dropdown.currentText() == ""
        ):
            self.align_image_button.setEnabled(False)
        else:
            self.align_image_button.setEnabled(True)

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
    MidplaneAligner,
    MidplaneEstimator,
)


class AlignMidplane(QWidget):
    """Widget to align the plane of symmetry to the midplane of the image."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        self._create_estimate_group()
        self._create_align_group()

    def _create_estimate_group(self):
        """Create the group of widgets concerned with estimating midplane
        points."""
        self.estimate_groupbox = QGroupBox("Estimate points along midplane")
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

        # Add dropdown to select axis
        self.select_axis_dropdown = QComboBox(parent=self.estimate_groupbox)
        self.select_axis_dropdown.addItems(["x", "y", "z"])
        self.estimate_groupbox.layout().addRow(
            "symmetry axis:", self.select_axis_dropdown
        )

        # Initialise button to estimate midplane points
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
        the midplane."""

        self.align_groupbox = QGroupBox("Align image to midplane")
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

        # Add button to align image to midplane
        self.align_image_button = QPushButton(
            "Align image", parent=self.align_groupbox
        )
        self.align_image_button.setEnabled(False)
        self.align_image_button.clicked.connect(self._on_align_button_click)
        self.align_groupbox.layout().addRow(self.align_image_button)

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
        """Estimate midplane points and add them to the viewer."""
        # Estimate 9 midplane points based on the selected mask
        mask_name = self.select_mask_dropdown.currentText()
        mask = self.viewer.layers[mask_name]
        axis = self.select_axis_dropdown.currentText()
        estimator = MidplaneEstimator(mask.data, symmetry_axis=axis)
        points = estimator.get_points()

        # Point layer attributes
        point_attrs = {
            "properties": {"label": list(range(9))},
            "face_color": "green",
            "symbol": "cross",
            "edge_width": 0,
            "opacity": 0.6,
            "size": 6,
            "ndim": mask.ndim,
            "name": "midplane points",
        }

        self.viewer.add_points(points, **point_attrs)
        self.refresh_dropdowns()
        # Move viewer to show z-plane of first point
        self.viewer.dims.set_point(0, points[0][0])
        # Enable "Select points" mode
        self.viewer.layers["midplane_points"].mode = "select"
        show_info("Please move all 9 estimated points exactly to the midplane")

    def _on_align_button_click(self):
        """Align image and mask to midplane and add them to the viewer."""
        image_name = self.select_image_dropdown.currentText()
        image_data = self.viewer.layers[image_name].data
        mask_name = self.select_mask_dropdown.currentText()
        mask_data = self.viewer.layers[mask_name].data
        points_name = self.select_points_dropdown.currentText()
        points_data = self.viewer.layers[points_name].data
        axis = self.select_axis_dropdown.currentText()

        aligner = MidplaneAligner(
            image_data,
            points_data,
            symmetry_axis=axis,
        )
        aligned_image = aligner.transform_image(image_data)
        self.viewer.add_image(aligned_image, name="aligned_image")
        aligned_mask = aligner.transform_image(mask_data)
        self.viewer.add_labels(aligned_mask, name="aligned_mask", opacity=0.5)
        aligned_halves = aligner.label_halves(aligned_image)
        self.viewer.add_labels(
            aligned_halves, name="aligned_halves", opacity=0.5
        )
        # Hide original image, mask, and points layers
        self.viewer.layers[image_name].visible = False
        self.viewer.layers[mask_name].visible = False
        self.viewer.layers[points_name].visible = False
        # Hide aligned mask layer
        self.viewer.layers["aligned_mask"].visible = False
        # Make aligner object accessible to other methods
        self.aligner = aligner

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

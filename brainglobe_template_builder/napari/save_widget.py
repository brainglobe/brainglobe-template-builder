from pathlib import Path

from brainglobe_utils.IO.image.save import save_as_asr_nii
from napari.layers import Image, Labels, Points
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from brainglobe_template_builder.io import save_3d_points_to_csv


class SaveFiles(QWidget):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        self._create_save_group()

    def _create_save_group(self):
        self.save_groupbox = QGroupBox("Save to output directory")
        self.save_groupbox.setLayout(QFormLayout())
        self.layout().addRow(self.save_groupbox)

        self._create_voxel_size_widget()
        self._create_save_path_widget()

        self.save_button = QPushButton(
            "Save selected layers", parent=self.save_groupbox
        )
        self.save_button.clicked.connect(self.save_selected_layers)
        self.save_groupbox.layout().addRow(self.save_button)

    def _create_voxel_size_widget(self):
        """Create 3 fields for entering the voxel size."""
        self.voxel_size_layout = QHBoxLayout()
        self.axis_0_size = QLineEdit()
        self.axis_1_size = QLineEdit()
        self.axis_2_size = QLineEdit()
        self.axis_0_size.setText("1")
        self.axis_1_size.setText("1")
        self.axis_2_size.setText("1")
        self.voxel_size_layout.addWidget(self.axis_0_size)
        self.voxel_size_layout.addWidget(self.axis_1_size)
        self.voxel_size_layout.addWidget(self.axis_2_size)

        self.save_groupbox.layout().addRow(
            "Voxel size (axes 0, 1, 2) in mm:", self.voxel_size_layout
        )

    def _create_save_path_widget(self):
        """Create a line edit and browse button for selecting a save path.

        The path has to be a valid directory path.
        """
        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("browse")
        self.browse_button.clicked.connect(self._open_save_dialog)

        self.path_layout = QHBoxLayout()
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.browse_button)
        self.save_groupbox.layout().addRow(
            "Output directory:", self.path_layout
        )

    def _open_save_dialog(self):
        """Select an existing directory path to save files to."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.AcceptMode(QFileDialog.AcceptSave)
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            self.path_edit.setText(path)

    def save_selected_layers(self):
        save_dir = self.path_edit.text()
        if not save_dir:
            return

        # Get voxel sizes
        try:
            vox_sizes = [
                float(self.axis_0_size.text()),
                float(self.axis_1_size.text()),
                float(self.axis_2_size.text()),
            ]
        except ValueError:
            show_info("Please enter valid voxel sizes in mm.")

        selected_layers = self.viewer.layers.selection
        saved_layer_names = []
        for layer in selected_layers:
            if isinstance(layer, Points):
                csv_path = f"{save_dir}/{layer.name}.csv"
                layer.save(csv_path)  # native napari save for points
                df_path = Path(f"{save_dir}/{layer.name}_df.csv")
                save_3d_points_to_csv(layer.data, df_path)  # with headers
                saved_layer_names.append(layer.name)
            elif isinstance(layer, Image) or isinstance(layer, Labels):
                tif_path = f"{save_dir}/{layer.name}.tif"
                nii_path = Path(f"{save_dir}/{layer.name}.nii.gz")
                # choose array dtype based on layer type
                dtype = "float32" if isinstance(layer, Image) else "uint8"
                layer.data = layer.data.astype(dtype)
                # native napari save to tif
                layer.save(tif_path)
                # save to nii
                save_as_asr_nii(
                    layer.data,
                    vox_sizes=vox_sizes,
                    dest_path=nii_path,
                )
                saved_layer_names.append(layer.name)
            else:
                raise UserWarning(
                    f"Layer {layer.name} is of type {type(layer)} and cannot "
                    "be saved. Only Image, Labels and Points layers are saved."
                )
        info_msg = "Saved layers: " + ", ".join(saved_layer_names)
        show_info(info_msg)

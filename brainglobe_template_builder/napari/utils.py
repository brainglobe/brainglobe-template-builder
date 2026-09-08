from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)


class DirPathWidget:
    """Widget to select a local directory path."""

    def __init__(self, group_box: QGroupBox, label: str):
        """
        Create a line edit and browse button for selecting a directory path.

        The chosen path has to be a valid directory path.

        Parameters
        ----------
        group_box : QGroupBox
            Group box to add the widget to.
        label : str
            Label for directory path.
        """
        self.group_box = group_box
        self.label = label
        self._create_dir_path_widget()

    def _create_dir_path_widget(self):
        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("browse")
        self.browse_button.clicked.connect(self._open_save_dialog)

        self.path_layout = QHBoxLayout()
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.browse_button)
        self.group_box.layout().addRow(self.label, self.path_layout)

    def _open_save_dialog(self):
        """Select an existing directory path."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.AcceptMode(QFileDialog.AcceptSave)
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            self.path_edit.setText(path)

    def get_dir_path(self):
        """Get chosen directory path."""
        return self.path_edit.text()


class VoxelSizeWidget(QWidget):
    """Widget for entering voxel sizes along axes 0, 1, and 2.

    The widget contains three line edits arranged horizontally, one per axis.
    Values are expected to be numeric and are interpreted as voxel sizes in mm.
    """

    def __init__(self, default_voxel_size="1", parent=None):
        """Initialise the voxel-size input widget.

        Parameters
        ----------
        default_voxel_size : str, optional
            Initial text value used for all three axis fields.
            Defaults to '1'.
        parent : QWidget, optional
            Parent Qt widget.
        """
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.axis_0_size = QLineEdit(default_voxel_size)
        self.axis_1_size = QLineEdit(default_voxel_size)
        self.axis_2_size = QLineEdit(default_voxel_size)

        for w in (self.axis_0_size, self.axis_1_size, self.axis_2_size):
            layout.addWidget(w)

    def get_voxel_sizes(self) -> tuple[float, float, float] | None:
        """Return voxel sizes entered for axes 0, 1, and 2.

        Returns
        -------
        tuple[float, float, float]
            Parsed voxel sizes for the three axes.

        Notes
        -----
        If parsing fails, an informational notification is shown.
        """
        try:
            return (
                float(self.axis_0_size.text()),
                float(self.axis_1_size.text()),
                float(self.axis_2_size.text()),
            )
        except ValueError:
            show_info("Please enter valid voxel sizes in mm.")
            return None

    def set_voxel_sizes(self, sizes: tuple[float, float, float]):
        """Set voxel sizes for axes 0, 1, and 2.

        Parameters
        ----------
        sizes : tuple[float, float, float]
            Values to set for axis 0, axis 1, and axis 2 respectively.
        """
        for w, v in zip(
            (self.axis_0_size, self.axis_1_size, self.axis_2_size), sizes
        ):
            w.setText(str(v))

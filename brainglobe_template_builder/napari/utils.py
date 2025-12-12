from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
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

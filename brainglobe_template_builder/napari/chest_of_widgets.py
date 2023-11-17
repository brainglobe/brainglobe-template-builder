from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from napari.viewer import Viewer
from qtpy.QtWidgets import QPushButton

from brainglobe_template_builder.napari.mask_widget import GenerateMask


class ChestOfDrawers(CollapsibleWidgetContainer):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__()
        self.add_widget(GenerateMask(napari_viewer, parent=self))
        self.add_widget(GenerateMask(napari_viewer, parent=self))

        random_button = QPushButton("Random button", parent=self)
        self.add_widget(random_button)

        self.add_widget(GenerateMask(napari_viewer, parent=self))

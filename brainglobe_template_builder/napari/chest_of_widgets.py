from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from napari.viewer import Viewer
from qtpy.QtWidgets import QPushButton

from brainglobe_template_builder.napari.mask_widget import GenerateMask


class ChestOfDrawers(CollapsibleWidgetContainer):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__()
        mask_widget_1 = GenerateMask(napari_viewer, parent=self)
        self.add_widget(mask_widget_1, collapsible=True, widget_title="Mask 1")

        random_button = QPushButton("Random button", parent=self)
        self.add_widget(random_button)

        mask_widget_2 = GenerateMask(napari_viewer, parent=self)
        self.add_widget(mask_widget_2, collapsible=True, widget_title="Mask 2")

from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from napari.viewer import Viewer

from brainglobe_template_builder.napari.mask_widget import CreateMask
from brainglobe_template_builder.napari.midline_widget import FindMidline


class PreprocWidgets(CollapsibleWidgetContainer):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__()

        self.add_widget(
            CreateMask(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Create mask",
        )

        self.add_widget(
            FindMidline(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Find midline",
        )

from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from napari.viewer import Viewer

from brainglobe_template_builder.napari.align_widget import AlignMidplane
from brainglobe_template_builder.napari.mask_widget import CreateMask


class PreprocWidgets(CollapsibleWidgetContainer):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__()

        self.add_widget(
            CreateMask(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Create mask",
        )

        self.add_widget(
            AlignMidplane(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Align midplane",
        )

        self.mask_widget, self.midplane_widget = self.collapsible_widgets
        # expand mask widget by default
        self.mask_widget.expand()
        # refresh dropdowns when midline widget is toggled
        self.midplane_widget.toggled.connect(
            self.midplane_widget.content().refresh_dropdowns
        )

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
        self._expand_mask_widget()

        self.add_widget(
            AlignMidplane(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Align midplane",
        )
        self._connect_midline_widget_toggle()

    def get_widgets(self):
        """Get all widgets in the container."""
        return [
            self.layout().itemAt(i).widget()
            for i in range(self.layout().count())
        ]

    def _expand_mask_widget(self):
        """Expand the mask widget."""
        mask_widget = self.get_widgets()[0]
        mask_widget.expand()

    def _connect_midline_widget_toggle(self):
        """Connect the toggle of the midline widget to the refresh of its
        dropdowns.
        """
        midline_widget = self.get_widgets()[1]
        midline_widget.toggled.connect(
            midline_widget.content().refresh_dropdowns
        )

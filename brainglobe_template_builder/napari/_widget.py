from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer

from brainglobe_template_builder.napari.align_widget import AlignMidplane
from brainglobe_template_builder.napari.mask_widget import CreateMask
from brainglobe_template_builder.napari.reorient_widget import Reorient
from brainglobe_template_builder.napari.save_widget import SaveFiles


class PreprocWidgets(CollapsibleWidgetContainer):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__()

        self.add_widget(
            Reorient(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Reorient to standard space",
        )

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

        self.add_widget(
            SaveFiles(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Save files",
        )

        (
            self.reorient_widget,
            self.mask_widget,
            self.midplane_widget,
            self.save_widget,
        ) = self.collapsible_widgets
        # expand first widget by default
        self.reorient_widget.expand()
        # refresh dropdowns when midline widget is toggled
        self.midplane_widget.toggled.connect(
            self.midplane_widget.content().refresh_dropdowns
        )

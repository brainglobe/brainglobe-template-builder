from brainglobe_template_builder.napari._widget import PreprocWidgets


def test_preproc_widgets(make_napari_viewer):
    """
    Test PreprocWidgets contains all relevant widgets with correct titles.
    """
    viewer = make_napari_viewer()
    preproc_widgets = PreprocWidgets(viewer)
    viewer.window.add_dock_widget(preproc_widgets)

    # Should be 4 widgets included in PreprocWidgets
    n_widgets = 4
    assert len(preproc_widgets.collapsible_widgets) == n_widgets
    assert (
        len(preproc_widgets.layout()) == n_widgets + 1
    )  # layout has one extra spacer item

    # Widgets should match the expected titles, and only the first should
    # be expanded (rest are collapsed by default)
    expected_widget_titles = [
        "Reorient to standard space",
        "Create mask",
        "Align midplane",
        "Save files",
    ]
    for i, title in enumerate(expected_widget_titles):
        widget = preproc_widgets.collapsible_widgets[i]
        assert widget.text() == title

        if i == 0:
            assert widget.isExpanded()
        else:
            assert not widget.isExpanded()

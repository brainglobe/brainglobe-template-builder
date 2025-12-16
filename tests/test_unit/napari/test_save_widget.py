from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from brainglobe_utils.IO.image.load import load_nii

from brainglobe_template_builder.napari.save_widget import SaveFiles


@pytest.fixture
def save_widget(tmp_path, make_napari_viewer):
    """
    Create a viewer, add the save widget, and return the widget.
    The viewer can be accessed using ``widget.viewer``.
    """
    viewer = make_napari_viewer()
    widget = SaveFiles(viewer)

    # voxel size of [0.01, 0.02, 0.03]
    widget.axis_0_size.setText("0.01")
    widget.axis_1_size.setText("0.02")
    widget.axis_2_size.setText("0.03")

    # Output to a temporary directory
    widget.output_dir_widget.path_edit.setText(str(tmp_path))

    viewer.window.add_dock_widget(widget)
    return widget


def test_save_stack(save_widget, stack):
    """Test the widget correctly saves ONLY the selected
    image to nifti."""
    viewer = save_widget.viewer

    # Add two images - we will save one of them
    viewer.add_image(stack, name="stack_to_save")
    viewer.add_image(stack, name="stack_NOT_to_save")

    # Select only one of the layers
    viewer.layers.selection.select_only(viewer.layers["stack_to_save"])

    save_widget.save_selected_layers()

    # Output directory should only contain saved 'stack_to_save'
    output_dir = Path(save_widget.output_dir_widget.get_dir_path())
    created_files = [file.name for file in output_dir.iterdir()]
    assert sorted(created_files) == [
        "stack_to_save.nii.gz",
        "stack_to_save.tif",
    ]

    saved_nifti = load_nii(output_dir / "stack_to_save.nii.gz", as_array=False)
    expected_zooms = (0.01, 0.02, 0.03)
    np.testing.assert_allclose(saved_nifti.header.get_zooms(), expected_zooms)
    np.testing.assert_array_equal(saved_nifti.get_fdata(), stack)


def test_save_points(save_widget, points):
    """Test the widget correctly saves a points layer."""

    viewer = save_widget.viewer
    viewer.add_points(points, name="points_to_save")
    viewer.layers.selection.select_only(viewer.layers["points_to_save"])

    save_widget.save_selected_layers()

    # Output directory should contain saved point csv files
    output_dir = Path(save_widget.output_dir_widget.get_dir_path())
    created_files = [file.name for file in output_dir.iterdir()]
    assert sorted(created_files) == [
        "points_to_save.csv",
        "points_to_save_df.csv",
    ]

    # Check saved csv files match input points
    expected_csv = pd.DataFrame(
        data=points, columns=["axis-0", "axis-1", "axis-2"]
    )
    assert pd.read_csv(
        output_dir / "points_to_save.csv", index_col="index"
    ).equals(expected_csv)

    expected_df_csv = pd.DataFrame(data=points, columns=["z", "y", "x"])
    assert pd.read_csv(output_dir / "points_to_save_df.csv").equals(
        expected_df_csv
    )

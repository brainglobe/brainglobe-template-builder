"""
This module is for finding the midline plane of a 3D image stack.

The image is loaded into napari, and the user is asked to annotate
at least 3 points on the midline. The midline plane is then fitted
to these points using a least squares regression. Finally, the
transformation matrix needed to rotate the image so that the midline
plane is centered is calculated and saved.
"""

# %%
from itertools import product
from pathlib import Path

import imio
import napari
import numpy as np
import pandas as pd
import skimage

# %%
# load tiff image into napari
data_dir = Path(
    "/Users/nsirmpilatze/Data/BlackCap/ants/template_v2/results_tiff"
)
image_path = data_dir / "template.tif"
image = imio.load_any(image_path.as_posix())

viewer = napari.view_image(image, name="image")

# %%
# Segment the brain from the background via thresholding
# and add brain mask to napari viewer

# first apply gaussian filter to image
image_smoothed = skimage.filters.gaussian(image, sigma=3)
viewer.add_image(
    image_smoothed,
    name="smoothed_image",
    visible=False,
)

# %%
# then apply thresholding to smoothed image
thresholded = skimage.filters.threshold_triangle(image_smoothed)
binary_smoothed = image_smoothed > thresholded
# Remove small objects
min_object_size = 500  # Define a minimum object size to keep (in pixels)
binary_smoothed = skimage.morphology.remove_small_objects(
    binary_smoothed, min_size=min_object_size
)
viewer.add_image(
    binary_smoothed,
    name="initial_binary",
    colormap="green",
    blending="additive",
    opacity=0.5,
)

# %%
# Erode the binary mask to remove the edges of the brain
# and the resulting mask to napari viewer as a label layer
eroded = skimage.morphology.binary_erosion(
    binary_smoothed, footprint=np.ones((5, 5, 5))
)
viewer.add_labels(
    eroded,
    name="brain_mask",
    opacity=0.5,
)
viewer.layers["initial_binary"].visible = False
viewer.layers["brain_mask"].selected_label = 1


# %%
# Define initial set of 9 midline points

# Find slices at 1/4, 2/4, and 3/4 of each dimension
slices = [int(dim / 4) * i for dim in image.shape for i in [1, 2, 3]]
z_slices, y_slices, x_slices = slices[0:3], slices[3:6], slices[6:9]
# Find points at the intersection the middle x slice with the y and z slices
grid_points = np.array(list(product(z_slices, y_slices, [x_slices[1]])))
print(grid_points)

# %%
# Add points to napari viewer
points_layer = viewer.add_points(
    grid_points,
    name="midline",
    face_color="#ffaa00",
    size=5,
    opacity=0.5,
    edge_color="#ff0000",
    edge_width=1.5,
    edge_width_is_relative=False,
)
# %%
# Go to the first z slice

# activate point layer
viewer.layers.selection.active_layer = points_layer
# change selection mode to select
viewer.layers.selection.mode = "select"
# go to the first z slice
viewer.dims.set_point(0, z_slices[0])

# %%
# save points to csv
midline_points = points_layer.data
midline_points = pd.DataFrame(midline_points, columns=["z", "y", "x"])
points_path = data_dir / "midline_points.csv"
midline_points.to_csv(points_path, index=False)

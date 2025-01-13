# %%
from pathlib import Path

import napari
from brainglobe_utils.IO.image import load_any, load_nii

# %%
# Define input/output paths

work_dir = Path.home() / "Dropbox/NIU/resources/figures/BlackCap_atlas_v1"
out_dir = work_dir / "img"
subject_id = "sub-BC73"
subject_image_name = f"{subject_id}_asym-brain.nii.gz"
subject_img_path = work_dir / "data" / subject_image_name

brainglobe_dir = Path.home() / ".brainglobe"
atlas_name = "oldenburg_blackcap_25um_v1.1"
average_img_path = brainglobe_dir / atlas_name / "reference.tiff"

# %%
# Load the images

subject_img = load_nii(subject_img_path, as_array=True, as_numpy=True)
print("Loaded subject image: ", subject_image_name)
print("Subject image shape: ", subject_img.shape)

template_img = load_any(average_img_path)
print("Loaded template image for ", atlas_name)
print("Template image shape: ", template_img.shape)

# %%
# Start napari viewer
viewer = napari.Viewer(ndisplay=3)

# Shared options for image layers
image_layer_opts = {
    "rendering": "attenuated_mip",
    "depiction": "volume",
}

# Define napari viewer angles for each camera view
camera_views = dict(
    front=(0, 0, 90),
    back=(0, 180, 90),
    top=(180, 0, 0),
    bottom=(180, 180, 0),
    left=(0, 90, 90),
    right=(0, -90, 90),
    oblique=(20, -40, 60),
)

# %%
# Add the average template image to the viewer
template_layer = viewer.add_image(
    template_img,
    name="average_template",
    experimental_clipping_planes=[
        {
            "position": [210, 0, 0],
            "normal": [1, 0, 0],
            "enabled": True,
        }
    ],
    **image_layer_opts,
)

# %%
# Take screenshot of the viewer at each pre-defined angle
for view_name, view_angles in camera_views.items():
    template_layer.experimental_clipping_planes[0].enabled = (
        True if view_name == "oblique" else False
    )
    template_layer.attenuation = 0.2 if view_name == "oblique" else 0.03

    viewer.camera.angles = view_angles
    viewer.screenshot(
        out_dir / f"{template_layer.name}_view-{view_name}.png",
        canvas_only=True,
    )
# Hide the template layer
template_layer.visible = False

# %%
# Add the subject image to the viewer
subject_layer = viewer.add_image(
    subject_img,
    name=subject_id,
    contrast_limits=[50, 2000],
    experimental_clipping_planes=[
        {
            "position": [205, 0, 0],
            "normal": [1, 0, 0],
            "enabled": True,
        }
    ],
    **image_layer_opts,
)

# %%
# Take screenshot of the viewer at each pre-defined angle
for view_name, view_angles in camera_views.items():
    subject_layer.experimental_clipping_planes[0].enabled = (
        True if view_name == "oblique" else False
    )
    subject_layer.attenuation = 0.2 if view_name == "oblique" else 0.03

    viewer.camera.angles = view_angles
    viewer.screenshot(
        out_dir / f"{subject_layer.name}_view-{view_name}.png",
        canvas_only=True,
    )

# %%

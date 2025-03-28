from pathlib import Path

import numpy as np

from brainglobe_template_builder.io import (
    load_tiff,
    save_as_asr_nii,
)
from brainglobe_template_builder.preproc.splitting import (
    generate_arrays_4template,
    get_right_and_left_slices,
    save_array_dict_to_nii,
)

# Define voxel size(in microns) of the lowest resolution image
lowres = 50

# Define voxel sizes in mm (for Nifti saving)
lowres_vox_sizes = [lowres * 1e-3] * 3  # in mm

project_folder_path = (
    "/mnt/d/template_wd/"  #  "/mnt/ceph/_projects/rat_atlas/derivatives"
)

# Get all subject IDs dynamically
subject_ids = [
    folder.name
    for folder in Path(project_folder_path).glob("*")
    if folder.is_dir() and folder.name.startswith("sub-")
]

# Initialize lists
rat_image_paths = []
rat_mask_paths = []
dimensions = []

for subject_id in subject_ids:
    rat_image_path = list(
        Path(project_folder_path).rglob(
            f"{subject_id}/{subject_id}_*_orig-asr_aligned.tif"
        )
    )
    rat_image_paths.extend(rat_image_path)

    rat_mask_path = list(
        Path(project_folder_path).rglob(
            f"{subject_id}/{subject_id}_*_orig-asr_label-brain_aligned.tif"
        )
    )
    rat_mask_paths.extend(rat_mask_path)

# Read images and store their dimensions
for img_path, mask_path in zip(rat_image_paths, rat_mask_paths):
    img_array = load_tiff(img_path)
    dimensions.append(img_array.shape)


# Compute max dimensions along each axis
max_z = max(dim[0] for dim in dimensions)
max_y = max(dim[1] for dim in dimensions)
max_x = max(dim[2] for dim in dimensions)

# Add 20 pixels to each of the maximum dimensions
max_z += 20
max_y += 20
max_x += 20

# Pad images to match the largest dimensions + 20 pixels all around
for img_path, mask_path in zip(rat_image_paths, rat_mask_paths):

    img = load_tiff(img_path)
    mask = load_tiff(mask_path)

    # Calculate how much padding is needed for each axis
    pad_z = max_z - img.shape[0]
    pad_y = max_y - img.shape[1]
    pad_x = max_x - img.shape[2]

    # Apply equal padding (divide the padding equally between start and end)
    pad_z_start, pad_z_end = pad_z // 2, pad_z - pad_z // 2
    pad_y_start, pad_y_end = pad_y // 2, pad_y - pad_y // 2
    pad_x_start, pad_x_end = pad_x // 2, pad_x - pad_x // 2

    # Apply padding to the image
    padded_img = np.pad(
        img,
        (
            (pad_z_start, pad_z_end),
            (pad_y_start, pad_y_end),
            (pad_x_start, pad_x_end),
        ),
        mode="constant",
        constant_values=0,
    )

    # Check if axes are even, if not add extra padding
    for axis in range(3):
        if padded_img.shape[axis] % 2 != 0:
            pad_width = [(0, 1) if i == axis else (0, 0) for i in range(3)]
            padded_img = np.pad(
                padded_img, pad_width, mode="constant", constant_values=0
            )

    # Apply padding to the mask
    padded_mask = np.pad(
        mask,
        (
            (pad_z_start, pad_z_end),
            (pad_y_start, pad_y_end),
            (pad_x_start, pad_x_end),
        ),
        mode="constant",
        constant_values=0,
    )

    # Check if axes are even, if not add extra padding
    for axis in range(3):
        if padded_mask.shape[axis] % 2 != 0:
            pad_width = [(0, 1) if i == axis else (0, 0) for i in range(3)]
            padded_mask = np.pad(
                padded_mask, pad_width, mode="constant", constant_values=0
            )

    # Generate new image and mask filename with '_padded'
    padded_filename = img_path.stem + "_padded.nii.gz"
    padded_filepath = img_path.parent / padded_filename
    padded_mask_filename = mask_path.stem + "_padded.nii.gz"
    padded_mask_filepath = mask_path.parent / padded_mask_filename

    save_as_asr_nii(padded_img, lowres_vox_sizes, padded_filepath)
    save_as_asr_nii(padded_mask, lowres_vox_sizes, padded_mask_filepath)

    # Flipping images

    # Create the 'flipped' folder if it doesn't exist
    flipped_folder = img_path.parent / "flipped"
    flipped_folder.mkdir(parents=True, exist_ok=True)

    # Flip the image and the mask along axis 2
    padded_flipped_img = np.flip(padded_img, axis=2)
    padded_flipped_mask = np.flip(padded_mask, axis=2)

    # Construct new filename for the flipped image
    flipped_filename = img_path.stem + "_padded_flipped.nii.gz"
    flipped_filepath = flipped_folder / flipped_filename
    flipped_mask_filename = mask_path.stem + "_padded_flipped.nii.gz"
    flipped_mask_filepath = flipped_folder / flipped_mask_filename

    save_as_asr_nii(padded_flipped_img, lowres_vox_sizes, flipped_filepath)
    save_as_asr_nii(
        padded_flipped_mask, lowres_vox_sizes, flipped_mask_filepath
    )

    # Splitting brains using brainglobe_template_builder.preproc.splitting

    # Slice into right and left hemispheres
    right_slices, left_slices = get_right_and_left_slices(padded_img)

    # Process images and masks
    subject = img_path.stem
    processed_arrays = generate_arrays_4template(
        subject, padded_img, padded_mask, pad=0
    )

    # Create mirrored folder
    mirrored_folder = img_path.parent / "mirrored"
    mirrored_folder.mkdir(parents=True, exist_ok=True)

    # Save processed arrays in the mirrored folder
    vox_sizes = lowres_vox_sizes
    save_array_dict_to_nii(processed_arrays, mirrored_folder, vox_sizes)

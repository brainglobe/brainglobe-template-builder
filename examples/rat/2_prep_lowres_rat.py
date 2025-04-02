from pathlib import Path

import ants
import numpy as np

from brainglobe_template_builder.io import load_any, save_as_asr_nii
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

# Bias field correction (to homogenise intensities)

# Initialize original image paths lists
rat_image_paths_orig = []
rat_mask_paths_orig = []

# Construct original image paths
for subject_id in subject_ids:
    rat_image_path_orig = list(
        Path(project_folder_path).rglob(
            f"{subject_id}/{subject_id}_*_orig-asr_aligned.nii.gz"
        )
    )
    rat_image_paths_orig.extend(rat_image_path_orig)

    rat_mask_path_orig = list(
        Path(project_folder_path).rglob(
            f"{subject_id}/{subject_id}_*_orig-asr_label-brain_aligned.nii.gz"
        )
    )
    rat_mask_paths_orig.extend(rat_mask_path_orig)

# Read original images with ants
for img_path_orig, mask_path_orig in zip(
    rat_image_paths_orig, rat_mask_paths_orig
):

    img_ants = ants.image_read(img_path_orig.as_posix())
    img_n4 = ants.n4_bias_field_correction(img_ants)
    img_n4_filename = Path(img_path_orig.with_suffix("").stem + "_N4.nii.gz")
    img_n4_path = Path(img_path_orig.parent / img_n4_filename)
    ants.image_write(img_n4, img_n4_path.as_posix())

    mask_ants = ants.image_read(mask_path_orig.as_posix())
    mask_n4 = ants.n4_bias_field_correction(mask_ants)
    mask_n4_filename = Path(mask_path_orig.with_suffix("").stem + "_N4.nii.gz")
    mask_n4_path = Path(mask_path_orig.parent / mask_n4_filename)
    ants.image_write(mask_n4, mask_n4_path.as_posix())


# Initialize bias corrected image paths lists
rat_image_paths = []
rat_mask_paths = []
dimensions = []

# Construct image paths of bias corrected images
for subject_id in subject_ids:
    rat_image_path = list(
        Path(project_folder_path).rglob(
            f"{subject_id}/{subject_id}_*_orig-asr_aligned_N4.nii.gz"
        )
    )
    rat_image_paths.extend(rat_image_path)

    rat_mask_path = list(
        Path(project_folder_path).rglob(
            f"{subject_id}/{subject_id}_*_orig-asr_label-brain_aligned_N4.nii.gz"
        )
    )
    rat_mask_paths.extend(rat_mask_path)


# Read bias corrected images and store their dimensions
for img_path, mask_path in zip(rat_image_paths, rat_mask_paths):
    img_array = load_any(img_path, as_numpy=True)
    dimensions.append(img_array.shape)


# Compute max dimensions along each axis
max_z = max(dim[0] for dim in dimensions)
max_y = max(dim[1] for dim in dimensions)
max_x = max(dim[2] for dim in dimensions)

# Add 20 pixels to each of the maximum dimensions
max_z += 20
max_y += 20
max_x += 20


for img_path, mask_path in zip(rat_image_paths, rat_mask_paths):

    img = load_any(img_path, as_numpy=True)
    mask = load_any(mask_path, as_numpy=True)

    # Padding images to match the largest dimensions + 20 pixels all around

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
    pad_width = [
        (0, 1) if dim % 2 != 0 else (0, 0) for dim in padded_img.shape
    ]
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
    pad_width = [
        (0, 1) if dim % 2 != 0 else (0, 0) for dim in padded_mask.shape
    ]
    padded_mask = np.pad(
        padded_mask, pad_width, mode="constant", constant_values=0
    )

    # Generate new image and mask filename with '_padded'
    padded_filename = img_path.with_suffix("").stem + "_padded.nii.gz"
    padded_filepath = img_path.parent / padded_filename
    padded_mask_filename = mask_path.with_suffix("").stem + "_padded.nii.gz"
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
    flipped_filename = img_path.with_suffix("").stem + "_padded_flipped.nii.gz"
    flipped_filepath = flipped_folder / flipped_filename
    flipped_mask_filename = (
        mask_path.with_suffix("").stem + "_padded_flipped.nii.gz"
    )
    flipped_mask_filepath = flipped_folder / flipped_mask_filename

    save_as_asr_nii(padded_flipped_img, lowres_vox_sizes, flipped_filepath)
    save_as_asr_nii(
        padded_flipped_mask, lowres_vox_sizes, flipped_mask_filepath
    )

    # Splitting brains using brainglobe_template_builder.preproc.splitting

    # Slice into right and left hemispheres
    right_slices, left_slices = get_right_and_left_slices(padded_img)

    # Process images and masks
    subject = img_path.with_suffix("").stem
    processed_arrays = generate_arrays_4template(
        subject, padded_img, padded_mask, pad=0
    )

    # Create mirrored folder
    mirrored_folder = img_path.parent / "mirrored"
    mirrored_folder.mkdir(parents=True, exist_ok=True)

    # Save processed arrays in the mirrored folder
    vox_sizes = lowres_vox_sizes
    save_array_dict_to_nii(processed_arrays, mirrored_folder, vox_sizes)

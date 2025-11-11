"""This script
- takes a hand-(ITK-snap)annotated right hemisphere of the blackcap template,
- smoothes the annotations (modal filter+binary erosion),
- mirrors both template and annotations to make them a symmetric brain
- saves the result as Nifti

This is a preparatory step for packaging.
"""

from pathlib import Path

import napari
import numpy as np
from brainglobe_utils.IO.image import load_nii, save_as_asr_nii
from scipy.ndimage import binary_erosion

from brainglobe_template_builder.postproc.annotation_filter import (
    apply_modal_filter,
)

show_in_napari = True

annotations_path = (
    Path.home() / "corrected_annotations/merged_unique_fixed_pallium.nii.gz"
)
annotations = load_nii(annotations_path, as_array=True, as_numpy=True).astype(
    np.uint16
)

filtered = annotations.copy()

filtered = apply_modal_filter(filtered)
has_label = filtered > 0
mirrored_has_label = np.flip(has_label, axis=2)
has_label = np.concatenate((has_label, mirrored_has_label), axis=2)
has_label = binary_erosion(has_label, iterations=1)
has_label = has_label[:, :, : has_label.shape[2] // 2]
filtered *= has_label

if __name__ == "__main__":
    reference_path = (
        Path.home() / "corrected_annotations/male-template_hemi-right.nii"
    )
    reference = load_nii(reference_path, as_array=True, as_numpy=True)
    save_as_asr_nii(
        np.concatenate((reference, np.flip(reference, axis=2)), axis=2),
        [0.025] * 3,
        Path.home() / "blackcap_male_template.nii.gz",
    )
    save_as_asr_nii(
        np.concatenate((filtered, np.flip(filtered, axis=2)), axis=2),
        [0.025] * 3,
        Path.home() / "blackcap_male_smoothed_annotations.nii.gz",
    )

    if show_in_napari:
        viewer = napari.Viewer()
        viewer.add_labels(annotations, name="original annotations")
        viewer.add_image(reference, name="template")
        viewer.add_labels(filtered, name="modal filtered annotations")
        napari.run()

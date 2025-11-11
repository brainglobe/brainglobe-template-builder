import numpy as np
from brainglobe_utils.IO.image import load_any

from brainglobe_template_builder.preproc.brightness import (
    correct_image_brightness,
)
from tests.fetch_test_data import corrected_image_path, image_path


def test_correct_image_brightness():
    """Test correct_image_brightness returns a very similar result to
    ants.n4_bias_field_correction."""

    input_image = load_any(image_path())
    spacing = [0.125, 0.5, 0.125]
    output_image = correct_image_brightness(input_image, spacing=spacing)

    ants_corrected_image = load_any(corrected_image_path())

    assert output_image.shape == ants_corrected_image.shape
    # Check largest pixel value difference between ants correction and
    # our own is 0.04
    # Note: the github actions windows runner needs this higher 0.04
    # threshold, but all others (as well as running locally on a windows
    # computer), can use a tighter 0.01 threshold
    assert np.absolute(ants_corrected_image - output_image).max() < 0.04

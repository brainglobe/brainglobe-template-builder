import numpy as np
from brainglobe_utils.IO.image import load_any

from brainglobe_template_builder.utils.brightness import (
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
    # Note: max difference values can differ slightly across platforms, but
    # the max relative difference should be below 1E-3 in all cases.
    np.testing.assert_allclose(output_image, ants_corrected_image, rtol=1e-03)

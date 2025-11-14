import numpy as np
import SimpleITK as sitk


def correct_image_brightness(
    input_image: np.ndarray, spacing: list[float]
) -> np.ndarray:
    """Correct image brightness using simple itk's
    N4BiasFieldCorrectionImageFilter.

    Default values are set to match those used in
    ants.n4_bias_field_correction (antspyx v0.6.1)

    Parameters
    ----------
    input_image : np.ndarray
        Input image to process.
    spacing : list
        Pixel size (in mm) for each axis - should
        be in same order as input_image axes.

    Returns
    ----------
    np.ndarray
        The processed image with float64 dtype.
    """

    # The output of GetLogBiasFieldAsImage is always float32,
    # so we need a float32 input image to avoid errors
    if input_image.dtype != np.float32:
        input_image = input_image.astype(np.float32, casting="same_kind")

    sitk_image = sitk.GetImageFromArray(input_image)

    # Conversion to sitk flips the axis order [z, y, x] -> [x, y, z],
    # so we must flip the spacing too
    sitk_image.SetSpacing(spacing[::-1])

    shrinkFactor = 4
    image_downsampled = sitk.Shrink(
        sitk_image, [shrinkFactor] * sitk_image.GetDimension()
    )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.SetConvergenceThreshold(1e-7)
    corrector.Execute(image_downsampled)

    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)
    corrected_image_full_resolution = sitk_image / sitk.Exp(log_bias_field)

    return sitk.GetArrayFromImage(corrected_image_full_resolution)

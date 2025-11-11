import numpy as np
import SimpleITK as sitk


def correct_image_brightness(
    input_image: np.ndarray, spacing: list[float]
) -> np.ndarray:
    """Correct image brightness using simple itk's
    N4BiasFieldCorrectionImageFilter.

    Default values are set to match those used in
    ants.n4_bias_field_correction.

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

    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(sitk_image)
    print("sitk_image max", filter.GetMaximum())
    print("sitk_image min", filter.GetMinimum())

    # Conversion to sitk flips the axis order [z, y, x] -> [x, y, z],
    # so we must flip the spacing too
    sitk_image.SetSpacing(spacing[::-1])

    shrinkFactor = 4
    image_downsampled = sitk.Shrink(
        sitk_image, [shrinkFactor] * sitk_image.GetDimension()
    )

    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(image_downsampled)
    print("image downsampled max", filter.GetMaximum())
    print("image downsampled min", filter.GetMinimum())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.SetConvergenceThreshold(1e-7)
    corrector.Execute(image_downsampled)

    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)

    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(log_bias_field)
    print("log bias field max", filter.GetMaximum())
    print("log bias field min", filter.GetMinimum())

    corrected_image_full_resolution = sitk_image / sitk.Exp(log_bias_field)

    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(corrected_image_full_resolution)
    print("corrected image max", filter.GetMaximum())
    print("corrected image min", filter.GetMinimum())

    return sitk.GetArrayFromImage(corrected_image_full_resolution)

from pathlib import Path

import SimpleITK as sitk


def correct_image_brightness(
    input_image_path: Path, output_image_path: Path
) -> None:
    """Correct image brightness using simple itk's
    N4BiasFieldCorrectionImageFilter.

    Default values are set to match those used in
    ants.n4_bias_field_correction.

    Parameters
    ----------
    input_image_path : Path
        Path to nifti image file
    output_image_path : Path
        Path to save processed nifti image file
    """

    # Read as float32 (same as default of ants.image_read)
    image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)

    shrinkFactor = 4
    image_downsampled = sitk.Shrink(
        image, [shrinkFactor] * image.GetDimension()
    )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.SetConvergenceThreshold(1e-7)
    corrector.Execute(image_downsampled)

    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected_image_full_resolution = image / sitk.Exp(log_bias_field)

    sitk.WriteImage(corrected_image_full_resolution, output_image_path)

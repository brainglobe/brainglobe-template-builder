## Results

### Population-Based Brain Templates

A brain template, or reference image, is a standard representation of brain anatomy that establishes a common coordinate system for localizing data and annotations. It also serves as a registration target for aligning individual brain images to this shared space, allowing for comparisons across different brains and with respect to defined brain regions.

The ideal template image should be of high quality—characterized by excellent resolution, signal-to-noise ratio, and contrast—and be representative of the population being studied. These qualities improve the accuracy and reliability of the registration process and subsequent analyses.

In the simplest scenario, a single brain image from a 'typical' healthy subject can serve as a template. However, this approach has limitations since the selected individual may have unique anatomical features that do not reflect the full range of variability within the population. In magnetic resonance imaging (MRI), there is a long-standing practice of creating population-based, unbiased average templates by iteratively aligning and averaging brain images from multiple subjects. This method has been employed to generate most state-of-the-art brain templates for both humans and animals, including the Allen Mouse Brain Common Coordinate Framework (CCF).

### Construction of a Population-Based Brain Template

To create a population-based brain template for the Eurasian blackcap, we utilized an iterative alignment and averaging approach. This method, commonly used in magnetic resonance imaging (MRI) to generate unbiased average templates, involves aligning multiple individual brain images to a common space and averaging them. By using a symmetric group-wise normalization (SyGN) algorithm, we ensured that the resulting template was both high-quality and representative of the population, providing a reliable reference for further analyses.

The process started with an initial template derived from one of the input images and involved iterative registration of each brain image to this template. Over several iterations, transformations were applied to adjust both intensity and shape, ultimately converging on an unbiased average representation. The final template maintained a high signal-to-noise ratio and contrast, preserving fine anatomical details and achieving left-right symmetry, which is crucial for certain analyses, such as estimating left-right anatomical differences.

### Overview of Blackcap Brain Template Construction

For this study, we constructed a brain template using 10 blackcap brain image stacks acquired via serial two-photon (STP) tomography. To enhance template quality, we selected imaging channels with optimal signal-to-noise ratio and contrast between gray and white matter structures. Symmetric input images were generated to create a left-right symmetric template, improving the utility of the template for population-level studies.

## Methods

### Preparation of Brain Images for Template Construction

We used the set of 10 blackcap brain image stacks obtained through serial two-photon (STP) tomography. We visually inspected the data to select channels with a good signal-to-noise ratio and clear contrast between gray and white matter structures. In 8/10 cases, we used the green autofluorescence channel, while in the remaining 2/10 cases, we used the blue channel.

We manually cropped each image to tightly enclose the brain tissue and downsampled them to an isotropic resolution of 25 µm. The images were reoriented to conform to BrainGlobe's ASR convention—placing the origin at the anterior superior right corner and ordering the axes as anterior-posterior, superior-inferior, and right-left. The reoriented images were then converted from TIFF to NIfTI format for compatibility with the Advanced Normalization Tools (ANTs) software suite.

To correct for intensity inhomogeneities, ANTs' N4 bias field correction function was applied. Brain masks were generated using a combination of functions from the scikit-image package. The images were blurred with a Gaussian kernel (standard deviation of 150 µm) and thresholded using the 'triangle' method. The largest connected component was selected from the thresholded mask, and binary closing was applied with a square footprint of 250 µm to eliminate small holes.

To ensure symmetry and exclude two damaged hemispheres, we virtually hemisected each brain image along the mid-sagittal plane. This was achieved by:

- Defining the mid-sagittal plane in one image by annotating multiple midline points and fitting a 2D plane.
- Rotating this image in 3D so its mid-sagittal plane aligned with the central sagittal plane of the image volume.
- Rigidly aligning each individual image to this rotated target using ANTs registration, ensuring consistent alignment of the mid-sagittal plane across subjects.
- Splitting aligned images into left and right hemispheres along the mid-sagittal plane, and discarding the two damaged hemispheres.
- Flipping the 18 intact hemispheres along the left-right axis to generate symmetric images.

These steps produced 18 symmetric brain images and their corresponding masks, ready for template construction.
The symmetry of the input images ensured the left-right symmetry of the resulting template, which is beneficial for specific applications. For instance, asymmetric templates cannot be used to estimate left-right anatomical differences in a population, as any observed differences may be confounded by the inherent asymmetry of the template itself.

### Iterative Template Construction with SyGN

The symmetric brain images and their masks served as inputs to the symmetric group-wise normalization (SyGN) algorithm for template construction, which can produce an unbiased average of the input images, in terms of both intensity and shape. The brain masks were employed during registration to focus computations within brain areas and exclude background.

SyGN is canonically implemented in the open-source toolkit Advanced Normalization Tools (ANTs) through the `antsMultivariateTemplateConstruction2.sh` script. Here we used an optimized implementation of this script, which offers enhanced features such as the ability to resume interrupted computations and automatic adjustment of registration parameters based on input image size and resolution.

Template construction was initialized using one of the input images as a starting template and executed in four stages with progressively higher-order transformation types: rigid (6 parameters), similarity (6 rigid parameters plus 1 uniform scaling), affine (12 parameters), and non-linear. The non-linear stage relied on  ANTs' symmetric diffeomorphic normalization, which is among the top-performing algorithms for non-linear image registration.

Each stage was passed through four iterations of the SyGN algorithm.
Briefly, each SyGN iteration involved the following steps:

1. Register each individual subject to the current template using linear and/or non-linear transformations (depending on the stage).
2. Transform the individual images to the current template space using the computed matrices and/or nonlinear deformation fields.
3. Calculate the voxel-wise mean of the transformed images and apply a sharpening filter to the resulting average intensity image.
4. Average the individual-to-template transformations from step 1 to obtain an average transformation, then invert it.
5. Apply the inverted average transformation from step 4 to the sharpened average intensity image from step 3. This step updates the template's shape to more closely reflect the average shape of the individual brains. The updated template becomes the new registration target for the next iteration.

The 16 average images from each stage and iteration were visually inspected to ensure convergence to a stable template. The final template, generated after the 4th iteration of the non-linear stage, was selected as the reference for the blackcap brain atlas.

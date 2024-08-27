## Generating an average brain template

### Population-based brain templates
A brain template, or reference image, is a standard representation of brain
anatomy that defines a common coordinate system for localising data and annotations of brain regions.  It also serves as a registration target for
aligning individual brain images to this space, where they can
be compared against each other and with respect to
the defined brain regions.
As such, the ideal template image benefits from being of
high quality - in terms of resolution, signal-to-noise ratio,
and contrast - and representative of the population being studied.
Such traits increase the accuracy and reliability of the registration process and the subsequent analyses.

In the simplest case, a single brain image of a 'typical' healthy subject
can serve as as template. However, the chosen individual may have atypical
anatomical features and in general, such a template cannot capture the full range of
variation in the population. In the field of magnetic resonance imaging (MRI),
there has been a long-standing practice of creating population-based unbiased average templates
by iteratively aligning and averaging images from multiple subjects.
Such approaches have been used to generate most state-of-the-art human and animal brain templates, including the Allen Mouse Brain Common Coordinate
Framweork (CCF).

### The symmetric group-wise normalization (SyGN) algorithm
Following this rationale, we set out to build a population average brain
template for the blackcap. Specifically, we followed the symmetric group-wise normalization (SyGN) method which has been successfully employed to construct MRI-based brain templates for multiple
species, including the rhesus macaque, the common marmoset, and the baboon.
SyGN is implemented in the open-source tookit Advanced Normalization
Tools (ANTs) as the `antsMultivariateTemplateConstruction2.sh`
script. The method relies on iterative applications of both linear (affine)
and non-linear transformations to register individual images. The non-linear
steps use ANTs' symmetric diffeomorphic normalization (SyN), which has been
ranked among the best performing algorithms for non-linear image registration.

Briefly, the SyGN method consists of the following steps (see also Fig. 1):
1. Start with an initial template image, which can be an average of a few
   subjects or a single subject.
2. Register each individual subject to the current template using a combination of linear affine transforms and non-linear symmetric diffeomorphic deformations.
3. Transform the individual images to the template space using the computed
   deformations and produce an average intensity image by taking the voxel-wise mean of the transformed images.
4. Average the individual-to-template transformations produced in step 2 to
   obtain an average deformation field, and invert it.
5. Apply the inverted average deformation field from step 4 to the average intensity image from step 3. This updates the template's shape towards the average shape of the individual brains. The updated template is then used as the new registraion target for the next iteration of the algorithm.

After a few iteration, the above process converges towards an unbiased average of the population (as represented by the input images) in terms of both intensity and shape.
The resulting template has high signal-to-noise ratio and contrast, while also preserving fine anatomical details present in the individual images.

Optionally, the resulting template could be made to be left-right symmetric,
which is useful for certain applications. For example, asymmetric templates
caanot be used for estimating left-right anatomical differences in the population, because the results cannot be disambiguated from the inherent asymmetry of the template itself.

### Building the blackcap brain template
We applied the SyGN method to the set of 10 blackap brain image stacks produced by
STP tomography for this study. We visually inspected the data to identify suitable channels that provided good signal-to-noise ratio as well as
good contrast between grey and white matter structures. In 8/10 cases, we ended up using the green auto-fluorescence channel, whereas in 2/10 we used the blue channel.

### Pre-processing

We manually cropped the images down to a box tightly enclosing the brain and downsampled them to an isotropic resolution of 25 µm.
We then used the brainglobe-space package to re-orient the downsampled images to the ASR convention used in BrainGlobe - i.e. origin at the anterior superior right corner of the image, and axis order anterior-posterior, superior-inferior, right-left. The re-oriented images were converted from tiff to NIfTI format so that they could be processed by ANTs.

We used ANTs' N4 bias field correction function to remove intensity inhomogeneities from the images, and generated brain masks using the sciki-image package. Specifically, we blurred the intensity-normalised images with a Gaussian kernel (standard deviation
of 150 um) and thresholded them using the 'triangle' method.
We then selected the largest connected component from the resulting
mask and applied binary closing to it (square footprint of 250um side length) to fill in any small holes.

The next step was to virtually hemisect the images by splitting
them along the mid-sagittal plane. This was necessary for two reasons.
Firstly, 2/10 subjects had one of the hemispheres damaged during sample
preparation, and we wanted to exclude these from the template. Secondly,
we want to supply the SyGN algorithm with perfectly symmetric images
as inputs, to guarantee that the resulting average is also left-right symmetric. We achieved both goals in the following way:
- We first used SyGN to create an initial template, using just 3 subjects.
- We manually defined the mid-sagittal plane in this initial template, by annotating multiple points along the midline and fitting a 2D plane to them.
- We rotated the initial template in 3D such that its mid-sagittal plane was
brought to the central sagittal plane ot the image volume.
- We then used ANTs registration to rigidly align the individual images to the rotated initial template. This procedure reliably brought the mid-sagittal plane of each individual brain to the central sagittal image
plane.
- We then split the aligned images to left and right hemispheres along the mid-sagittal plane, and discared the 2 damaged hemispheres.
- Each of the 18 intact hemispheres was flipped along the left-right axis to create a virtually symmetric image.

The above procedured produced 18 perfectly symmetric images, which we used as input to the SyGN algorithm. We ran the algorithm for 4 iterations, using one of the images as the initial registration target.

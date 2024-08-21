## Generating an average brain template

A brain template, or reference image, is a standard representation of brain
anatomy that defines a common coordinate system for comparing
brain images across subjects, and in which annotation of brain regions are
defined.  It also serves as a registration target for
aligning individual brain images to the common atlas space, where they can
be compared against each other and with respect to the defined brain regions.
As such, the ideal template image benefits from being of high quality
- i.e. high resolution, signal-to-noise ratio, and contrast - and
representative of the population being studied.

In the simplest case, a single brain image of a 'typical' healthy subject
can be used as a template. However,the chosen individual may have atypical
anatomical features and the template may not capture the full range of
variation in the population. In the field of magnetic resonance imaging (MRI),
there has been a long-standing tradition of creating population-based templates
by aligning and averaging images from multiple subjects.
Several approaches have been employed to produce an unbiased average of the
population, both in terms of intensity values and brain shape, while also
avoiding blurring to preserve the fine anatomical details present in the
individual images. Most state-of-the-art templates are generated using
variants of this approach, including the Allen Mouse Brain Common Coordinate
Framweork (CCF).

For the blackap brain template, we decided to build a population average
template using the symmetric group-wise normalization (SyGN) method which
has been recently employed to construct MRI-based brain templates for multiple
species, including the rhesus macaque, the common marmoset, and the baboon.
SyGN is implemented in the open-source tookit Advanced Normalization
Tools (ANTs) as the `antsMultivariateTemplateConstruction2.sh`
script. The method relies on iterative applications of both linear affine
and non-linear transformations to register individual images. The non-linear
steps use ANTs' symmetric diffeomorphic normalization (SyN), which has been
ranked among the best performing algorithms for non-linear image registration.

The SyGN method consists of the following steps.
1. Start with an initial template image, which can be an average of a few
   subjects or a single subject.
2. Register each individual subject to the current template using a combination
of linear affine transforms and non-linear symmetric diffeomorphic deformations.
1. Transform the individual images to the template space using the computed
   transformations and average them.
2. Average the individual-to-template transformations and invert the resulting
defomation field.
1. Apply the inverted deformation field to the average image to produce an
updated template, and repeat the process until convergence is reached.

Each template update step, i.e the application of the inverted average
deformation field, modifies the template towards an intermediate of the
individual images, in both appearance and shape.

The above algorithm is implemented in

The resulting template is then used as the new template in the next iteration
of the algorithm. This process is repeated until convergence is reached.

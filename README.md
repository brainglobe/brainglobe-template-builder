[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
![CI](https://img.shields.io/github/actions/workflow/status/brainglobe/brainglobe-template-builder/test_and_deploy.yml?label=CI)
[![codecov](https://codecov.io/gh/brainglobe/brainglobe-template-builder/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/brainglobe/brainglobe-template-builder)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# brainglobe-template-builder
Build unbiased anatomical templates from individual images

- [Overview](#overview)
- [Scope](#scope)
- [Status](#status)
- [Background](#background)
  - [On templates and atlases](#on-templates-and-atlases)
  - [Single-subject vs population templates](#single-subject-vs-population-templates)
  - [Template construction with ANTs](#template-construction-with-ants)
- [License](#license)
- [Template](#template)

## Overview
brainglobe-template-builder aims to assist researchers in constructing unbiased anatomical reference images, commonly known as templates.

The primary goal is to create 'average' brain templates from a set of individual high-resolution brain images acquired across multiple subjects. That said, the under- 
lying algorithms and code should be applicable to any other organ for which digital 3D images are available.

We aim to primarily support imaged produced by 3D volumetric microscopy, such as serial two-photon tomography (STPT) and light-sheet microscopy (LSM).

## Scope
The main aims of brainglobe-template-builder are to:
- provide a user-friendly and accessible Python-based interface to the [optimised ANTs template construction pipeline](#template-construction-with-ants)
- support the ingestion of 3D volumetric microscopy images, such as STPT and LSM
- produce templates that are compatible with the [BrainGlobe ecosystem](https://brainglobe.info/), especially the [BrainGlobe Atlas API](https://brainglobe.info/documentation/bg-atlasapi/index.html)

## Status
> **Warning**
> - 🏗️ The package is currently in early development. Stay tuned ⌛
> - It is not sufficiently tested to be used for scientific analysis
> - The interface is subject to changes.
> 

## Background
### On templates and atlases
In the context of brain imaging, a *template* is an image that serves as a standard reference for brain anatomy. The terms *template* and *reference image* are used interchangeably.

Templates are typically used for registering (aligning) multiple individuals into a common coordinate space. This space can be either volumetric (3D coordinates system), surface-based (mesh), or a combination of both. Here, we refer to the volumetric case, unless otherwise specified.

For human MRI, the [MNI template](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009) serves as the community standard volumetric template. Its integration into most major software packages makes it easy for researchers to register and report
their results to MNI space. For neuroscientists working with mice, the [Allen Mouse Brain Atlas](https://mouse.brain-map.org/static/atlas) is playing a similar role. This facilitates data-sharing, cross-study comparisons and meta-analyses.

An *atlas* is a template that has been annotated with regions of interest - variably referred to as labels, parcellations, or annotations. The benefit of using an atlas is that it allows researchers to know which brain region they are looking at, and to extract quantitative information from that region. A prerequisite, is that their data is registered
to the corresponding template - i.e. the reference space in which the atlas labels are defined.

This whole process relies on the quality of the template image, the accuracy of the registration, and the quality of the atlas labels. The cornerstone is the availability of a high-quality template image - i.e. a 3D image with good contrast, high resolution, and representative of the population of interest. Such a template can improve the accuracy of the registration as well as ease the annotation of atlas labels.

brainglobe-template-builder aims to assist researchers in constructing such high-quality templates.

### Single-subject vs population templates
In theory, a brain of a single individual could serve as the template, which would require the acquisition of only one high quality image. These are referred to as *single-subject* templates and they may be sufficient, or even desirable in some applications.

However, a single subject, even if carefully selected to be a *typical* healthy individual, may be at an extreme tail of the normal distribution in some brain regions. Moreover, a single subject template cannot represent the anatomical variability in the population.

The solution is to build templates from multiple subjects, by aligning them to a common space and averaging the result. These are referred to as *population* templates.

The construction of population templates is a well-established process in human MRI ([Avants et al., 2010](https://www.sciencedirect.com/science/article/pii/S1053811909010611); [Fonov et al., 2011](https://www.sciencedirect.com/science/article/pii/S1053811910010062#s0010)) and has been also applied to animal brains, such as the aforementioned Allen Mouse Brain Common Coordinate Framework([Wang et al., 2020](https://www.sciencedirect.com/science/article/pii/S0092867420304025#bib21)) and the [NIMH Macaque Template](https://www.sciencedirect.com/science/article/pii/S1053811921002743?via%3Dihub).

### Template construction with ANTs
brainglobe-template-builder relies on the population template construction algorithm implemented in [ANTs (Advanced Normalisation Tools)](http://stnava.github.io/ANTs/) - a widely used software package for image registration and segmentation.

The ANTs unbiased template construction method consists of the iterative application of two major stages, the registration stage, and the template updating stage. You can find a more detailed description of the algorithm [here](https://github.com/ANTsX/ANTs/issues/520).

In particular, brainglobe-template-builder uses an optimised version of the [antsMultivariateTemplateConstruction2.sh](https://github.com/ANTsX/ANTs/blob/master/Scripts/antsMultivariateTemplateConstruction2.sh) script. This optimised implementation was developed by the [CoBra lab](https://www.cobralab.ca/) and is available through the [optimized_antsMultivariateTemplateConstruction GitHub repository](https://github.com/CoBrALab/optimized_antsMultivariateTemplateConstruction/tree/master).

## License
⚖️ [BSD 3-Clause](./LICENSE)

## Template
This package layout and configuration (including pre-commit hooks and GitHub actions) have been copied from the [python-cookiecutter](https://github.com/neuroinformatics-unit/python-cookiecutter) template.




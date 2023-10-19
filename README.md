[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
![CI](https://img.shields.io/github/actions/workflow/status/brainglobe/brainglobe-template-builder/test_and_deploy.yml?label=CI)
[![codecov](https://codecov.io/gh/brainglobe/brainglobe-template-builder/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/brainglobe/brainglobe-template-builder)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# brainglobe-template-builder
Build unbiased anatomical templates from individual images

- [Overview](#overview)
- [Installation](#installation)
- [Background](#background)
  - [On templates and atlases](#on-templates-and-atlases)
  - [Single-subject vs population templates](#single-subject-vs-population-templates)
  - [Template construction with ANTs](#template-construction-with-ants)
- [License](#license)
- [Package blueprint](#package-blueprint)

## Overview

`brainglobe-template-builder` provides a streamlined process to create unbiased anatomical reference images, or templates, from multiple high-resolution brain images. While primarily designed for brain imaging, its versatility extends to any organ with available 3D digital images, especially those produced by 3D volumetric microscopy like serial two-photon tomography (STPT) and light-sheet microscopy (LSM).

`brainglobe-template-builder` aims to:
- Offer an intuitive Python interface to the [optimised ANTs template construction pipeline](#template-construction-with-ants).
- Support 3D volumetric microscopy images, such as STPT and LSM.
- Generate templates compatible with the [BrainGlobe ecosystem](https://brainglobe.info/), especially the [BrainGlobe Atlas API](https://brainglobe.info/documentation/bg-atlasapi/index.html).

> **Warning**
> - Early development phase. Stay tuned
> - Interface may undergo changes.

## Installation

> **Warning**
> - [ANTs](http://stnava.github.io/ANTs/), which [we depend on](#template-construction-with-ants), is a large package. The installation may take a while.
> - ANTs is only available for Linux and macOS. If you are on Windows, you can use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

We recommend installing `brainglobe-template-builder` within a [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/index.html) environment. Instructions assume `conda` usage, but `mamba`/`micromamba` are interchangeable.

Use the provided `environment.yaml` file to create a new environment.

```sh
git clone https://github.com/brainglobe/brainglobe-template-builder
cd brainglobe-template-builder
conda env create -f environment.yaml -n template-builder
conda activate template-builder
```

We have called the environment `template-builder`, but you can choose any name you like.

To install the latest development version of `brainglobe-template-builder`,

```sh
pip install -e .[dev]
```
For zsh users (default shell on macOS):

```sh
pip install -e '.[dev]'
```

## Background

### On templates and atlases

In brain imaging, a *template* serves as a standard reference for brain anatomy, often used interchangeably with the term *reference image*. By aligning multiple brain images to a common template, researchers can standardize their data, facilitating easier data-sharing, cross-study comparisons, and meta-analyses.

An *atlas* elevates this concept by annotating a template with regions of interest, often called labels or parcellations. With an atlas, researchers can pinpoint specific brain regions and extract quantitative data from them.

The entire process, from registration to data extraction, hinges on the quality of the template image. A high-quality template can significantly improve registration accuracy and the precision of atlas label annotations.

The aim of `brainglobe-template-builder` is to assist researchers in constructing such high-quality templates.

### Single-subject vs population templates

Templates can be derived in two primary ways. A *single-subject* template is based on the brain of one individual. While this approach is simpler and may be suitable for some applications, it risks being unrepresentative, as the chosen individual might have unique anatomical features. On the other hand, *population* templates are constructed by aligning and averaging brain images from multiple subjects. This method captures the anatomical variability present in a population and reduces biases inherent in relying on a single subject. Population templates have become the standard in human MRI studies and are increasingly being adopted for animal brain studies.

### Template construction with ANTs
`brainglobe-template-builder` leverages the power of [ANTs (Advanced Normalisation Tools)](http://stnava.github.io/ANTs/), a widely used software suite for image registration and segmentation.

ANTs includes a template construction piepline - implemented in the [antsMultivariateTemplateConstruction2.sh](https://github.com/ANTsX/ANTs/blob/master/Scripts/antsMultivariateTemplateConstruction2.sh) script - that iteratively aligns and averages multiple images to produce an unbiased population template (see [this issue](https://github.com/ANTsX/ANTs/issues/520) for details).

An [optimsed implementation of the above pipeline](https://github.com/CoBrALab/optimized_antsMultivariateTemplateConstruction/tree/master), developed by the [CoBra lab](https://www.cobralab.ca/), lies at the core of the `brainglobe-template-builder`'s functionality.

## License
⚖️ [BSD 3-Clause](https://opensource.org/license/bsd-3-clause/)

## Package blueprint
This package layout and configuration (including pre-commit hooks and GitHub actions) have been copied from the [python-cookiecutter](https://github.com/neuroinformatics-unit/python-cookiecutter) template.

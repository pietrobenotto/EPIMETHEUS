# EPIMETHEUS

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://your-package.readthedocs.io)

*Effective Psf Inferred from Mosaics using an Estimation Technique for Heterogeneous Exposures to get a Unique PSF across the Sensor field*

## Features

- Extraction: Source extraction and star selection in one or multiple images
- Rotation: Possibility to automatically retrieve the most common orientation of the exposures composing a mosaic
- Selection: Both automatic star selection based on their rotation and graphical tool for the manual selection.
- Stacking: ePSF generation using iterative alignment, MAD rejection algorithm, and providing ePSFs with different orientations

Each star is manipulated only once from the initial cutout to the final ePSF to minimize the blurring effect of multiple image transformation

## Installation

### Prerequisites

This package requires [SExtractor](https://github.com/astromatic/sextractor) to be installed on your system. The easiest way to install it is via Conda:

```bash
conda install -c conda-forge astromatic-source-extractor
```

### Installing the Package

Install using pip:

```bash
pip install .
```


## Quick Start

```python
import epimetheus as epi
```

A working example in a jupyter notebook is provided in examples folder

## License

This project is licensed under the MIT License.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{Benotto2026,
  author = {Pietro Benotto},
  title = {EPIMETHEUS: Effective PSF Inferred from Mosaics},
  year = {In prep.},
}
```

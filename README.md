# EPIMETHEUS

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*Effective PSF Inferred from Mosaics using an Estimation Technique for Heterogeneous Exposures to get a Unique PSF across the Sensor field*

## Features

- Extraction: Source extraction and star selection in one or multiple images
- Rotation: Possibility to automatically retrieve the most common orientation of the exposures composing a mosaic
- Selection: Both automatic star selection based on their rotation and a graphical tool for manual selection.
- Stacking: ePSF generation using iterative alignment, MAD rejection algorithm, and providing ePSFs with different orientations

Each star is manipulated only once from the initial cutout to the final ePSF to minimise the blurring effect of multiple image transformations.

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

Two working examples of EPIMETHEUS are provided in the examples folder

## HST and JWST ePSFs release

In the ePSFs folder, we release ePSFs for many HST filters and all JWST/NIRCam filters. We computed the ePSF using the data from the COSMOS, UDS, GOODS-S and Abell 2744 fields simultaneously.

## License

This project is licensed under the MIT License.

## Citation

If you use this package in your research, please cite:

```bibtex
@ARTICLE{Benotto:2026,
       author = {{Benotto}, Pietro and {Vulcani}, Benedetta and {Altomonte}, Vittoria},
        title = "{EPIMETHEUS: Effective PSF inferred from mosaics}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics, Astrophysics of Galaxies},
         year = 2026,
        month = sep,
          eid = {arXiv:2609.26879},
        pages = {arXiv:2609.26879},
archivePrefix = {arXiv},
       eprint = {2609.26879},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260926879B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

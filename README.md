# EPIMETHEUS

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*Effective PSF Inferred from Mosaics using an Estimation Technique for Heterogeneous Exposures to get a Unique PSF across the Sensor field*

For any information or help in using the software, please contact pietro.benotto@inaf.it

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
@ARTICLE{2026A&A...714A.511B,
       author = {{Benotto}, Pietro and {Vulcani}, Benedetta and {Altomonte}, Vittoria},
        title = "{EPIMETHEUS: Effective point spread function inferred from mosaics}",
      journal = {\aap},
     keywords = {techniques: image processing, techniques: photometric},
         year = 2026,
        month = oct,
       volume = {714},
          eid = {A511},
        pages = {A511},
          doi = {10.1051/0004-6361/202660382},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026A&A...714A.511B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

# astro_utils

A Python package for astronomical spectroscopy analysis, originally designed for the MUSE catalogues presented in Richard et al. 2021 (R21): https://arxiv.org/abs/2009.09784

## Features

- Spectral line fitting with support for single and double-peaked profiles
- Asymmetric Gaussian models for Lyman-alpha emission
- Support for damped Lyman-alpha absorption (DLA) profiles
- Automated masking of sky lines and contaminating features
- Monte Carlo bootstrap error estimation with correlated noise support
- Integration with R21 MUSE catalogs
- Velocity/wavelength conversion utilities
- Stacking of multiply-lensed sources
- Stacking of multiple spectra and lines

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/astro_utils.git
cd astro_utils

# Install in development mode
pip install -e .
```

### Dependencies

- numpy
- scipy
- astropy
- matplotlib
- mpdaf
- error_propagation
- beautifulsoup4
- requests

## Configuration

### Quick Start

Set the base data directory where your MUSE catalogs and spectra are stored:

```bash
export R21_DATA_DIR="/path/to/your/data"
```

Optionally, set the R21 data URL (you'll be prompted if not set):

```bash
export R21_URL="https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/"
```

### Using a .env File

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your paths:
   ```bash
   ASTRO_DATA_DIR=/path/to/your/data
   R21_URL=https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/
   ```

3. Load the environment in Python:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   
   from astro_utils import spectroscopy, fitting
   ```

### Environment Variables

- `ASTRO_DATA_DIR` - Base directory for all data (default: `~/.astro_data`)
- `R21_URL` - Base URL for R21 data downloads
- `R21_SPECTRA_DIR` - Override for spectra location
- `R21_CATALOG_DIR` - Override for catalog location
- `SOURCE_SPECTRA_DIR` - Override for source spectra location

See [CONFIG.md](CONFIG.md) for detailed configuration documentation.

## Usage

### Loading Spectra

```python
from astro_utils import spectroscopy as spectro

# Load R21 spectrum
spec = spectro.load_r21_spec('A2744', '1234', 'E', 'weight_skysub')

# Load aperture spectrum
spec = spectro.load_aper_spec('A2744', '1234', 'E', '2fwhm')
```

### Fitting Emission Lines

```python
from astro_utils import fitting

# Fit a single emission line
result = fitting.fit_line(
    wavelength, spectrum, errors, 
    linename='CIV1548',
    initial_guesses={'LPEAK': 6850, 'FLUX': 100, 'FWHM': 3.5},
    bounds={'LPEAK': (6845, 6855)}
)

# Fit Lyman-alpha (automatic single/double peak detection)
lya_result = fitting.fit_lya_complete(
    wavelength, spectrum, errors, row, 
    width=50, mc_niter=500
)
```

### Velocity Conversions

```python
from astro_utils import spectroscopy as spectro

# Convert wavelength to velocity
vel = spectro.wave2vel(observed_wavelength=6850, 
                       rest_wavelength=1215.67, 
                       redshift=4.64)

# Convert velocity to wavelength
wave = spectro.vel2wave(vel=200, restLambda=1215.67, z=4.64)
```

### Plotting

```python
from astro_utils import plotting
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plotting.plotline(
    iden='1234', clus='A2744', idfrom='E',
    wln=6850, ax_in=ax, width=50,
    spec_source='R21'
)
plt.show()
```

## Module Overview

### `spectroscopy.py`
- Spectrum loading and preprocessing
- Wavelength/velocity conversions
- MUSE LSF utilities

### `fitting.py`
- Line fitting routines
- Monte Carlo bootstrap error estimation
- Masking utilities
- Lyman-alpha profile fitting

### `models.py`
- Gaussian and asymmetric Gaussian models
- Voigt profiles
- DLA absorption models

### `catalogue_operations.py`
- R21 catalog loading and downloading
- Counterpart matching
- Result insertion

### `constants.py`
- Physical constants
- Line wavelength dictionary
- Doublet definitions

### `plotting.py`
- Spectrum visualization
- Model overlay plotting

## Data Structure

The package expects the following directory structure (created automatically):

```
$ASTRO_DATA_DIR/
├── muse_catalogs/
│   ├── catalogs/          # R21 catalog FITS files
│   │   ├── A2744_v1.0.fits
│   │   ├── A2744_v1.0_lines.fits
│   │   └── ...
│   └── spectra/           # R21 spectra
│       ├── A2744/
│       │   ├── spec_X1234_weight_skysub.fits
│       │   └── ...
│       └── ...
└── source_spectra/        # Aperture spectra
    ├── A2744/
    │   ├── idX1234_2fwhm_spec.fits
    │   └── ...
    └── ...
```

## Examples

See the `notebooks/` directory for Jupyter notebook examples demonstrating:
- Spectral line fitting workflows
- Lyman-alpha profile analysis
- Catalog operations
- Custom plotting

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this package in your research, please cite:
- Richard et al. (2021) for the MUSE catalogs
- Your publication using this software

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

This package uses data from the Richard et al. (2021) MUSE lensing cluster survey.

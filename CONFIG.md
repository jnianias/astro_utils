# Configuration Guide for astro_utils

This document describes the environment variables used by `astro_utils` for configuring data paths and URLs.

## Quick Start

The simplest way to use `astro_utils` is to set the base data directory:

```bash
export ASTRO_DATA_DIR="/path/to/your/data"
```

All other paths will be automatically derived from this base directory using the following structure:

```
$ASTRO_DATA_DIR/
├── muse_catalogs/
│   ├── catalogs/          # R21 catalog files
│   └── spectra/           # R21 spectra files
│       ├── A2744/
│       ├── MACS0416/
│       └── ...
├── muse_data/             # MUSE data cubes
│   ├── A2744/
│   │   └── cube/          # Cube FITS files
│   ├── MACS0416/
│   │   └── cube/
│   └── ...
└── source_spectra/        # Aperture spectra files
    ├── A2744/
    ├── MACS0416/
    └── ...
```

## Environment Variables

### Required Variables

None of the environment variables are strictly required. If not set, the package will use sensible defaults.

### Optional Variables

#### `ASTRO_DATA_DIR`
- **Description**: Base directory for all astronomical data
- **Default**: `~/.astro_data` (in your home directory)
- **Example**: `export ASTRO_DATA_DIR="/data/astronomy"`

#### `R21_URL`
- **Description**: Base URL for downloading R21 spectra and catalogs
- **Default**: Will prompt interactively when needed
- **Example**: `export R21_URL="https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/"`
- **Note**: The package will automatically append the appropriate path (e.g., `A2744_final_catalog/spectra/`)

#### `R21_SPECTRA_DIR`
- **Description**: Override for the R21 spectra directory (overrides `ASTRO_DATA_DIR` setting)
- **Default**: `$ASTRO_DATA_DIR/muse_catalogs/spectra`
- **Example**: `export R21_SPECTRA_DIR="/custom/path/to/spectra"`

#### `R21_CATALOG_DIR`
- **Description**: Override for the R21 catalog directory (overrides `ASTRO_DATA_DIR` setting)
- **Default**: `$ASTRO_DATA_DIR/muse_catalogs/catalogs`
- **Example**: `export R21_CATALOG_DIR="/custom/path/to/catalogs"`

#### `SOURCE_SPECTRA_DIR`
- **Description**: Override for the source spectra directory (overrides `ASTRO_DATA_DIR` setting)
- **Default**: `$ASTRO_DATA_DIR/source_spectra`
- **Example**: `export SOURCE_SPECTRA_DIR="/custom/path/to/source_spectra"`

#### `MUSE_CUBE_DIR`
- **Description**: Override for the MUSE data cube directory (overrides `ASTRO_DATA_DIR` setting)
- **Default**: `$ASTRO_DATA_DIR/muse_data`
- **Example**: `export MUSE_CUBE_DIR="/custom/path/to/muse_data"`
- **Note**: Each cluster's cubes should be in subdirectories: `$MUSE_CUBE_DIR/{cluster}/cube/*.fits`

## Setting Environment Variables

### Temporary (Current Session Only)

```bash
export ASTRO_DATA_DIR="/path/to/your/data"
export R21_URL="https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/"
```

### Permanent (Add to Shell Configuration)

Add these lines to your shell configuration file:
- **Bash**: `~/.bashrc` or `~/.bash_profile`
- **Zsh**: `~/.zshrc`
- **Fish**: `~/.config/fish/config.fish`

Example for Bash/Zsh:
```bash
# Add to ~/.bashrc or ~/.zshrc
export ASTRO_DATA_DIR="/path/to/your/data"
export R21_URL="https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/"
```

After editing, reload your configuration:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Using a .env File (Python)

You can also use a `.env` file with the `python-dotenv` package:

1. Create a `.env` file in your project directory:
```bash
ASTRO_DATA_DIR=/path/to/your/data
R21_URL=https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/
```

2. Load it in your Python script:
```python
from dotenv import load_dotenv
load_dotenv()

# Now import astro_utils
from astro_utils import spectroscopy, fitting
```

## Data Directory Structure

The package expects the following directory structure (created automatically as needed):

```
$ASTRO_DATA_DIR/
├── muse_catalogs/
│   ├── catalogs/
│   │   ├── A2744_v1.0.fits
│   │   ├── A2744_v1.0_lines.fits
│   │   ├── MACS0416_v1.1.fits
│   │   ├── MACS0416_v1.1_lines.fits
│   │   └── ...
│   └── spectra/
│       ├── A2744/
│       │   ├── spec_X1234_weight_skysub.fits
│       │   └── ...
│       ├── MACS0416/
│       │   └── ...
│       └── BULLET/
│           └── ...
├── muse_data/
│   ├── A2744/
│   │   └── cube/
│   │       ├── DATACUBE-A2744.fits
│   │       └── ...
│   ├── MACS0416/
│   │   └── cube/
│   │       └── ...
│   └── ...
└── source_spectra/
    ├── A2744/
    │   ├── idX1234_2fwhm_spec.fits
    │   └── ...
    ├── MACS0416/
    │   └── ...
    └── ...
```

## Migration from Hardcoded Paths

If you were previously using the hardcoded paths:
- **Old path**: `/media/james/63C4C5633F1EAE9F/phd/lya_outflows/muse_catalogs/`
- **New approach**: Set `ASTRO_DATA_DIR="/media/james/63C4C5633F1EAE9F/phd/lya_outflows"` or move your data to the new default location.

## Troubleshooting

### Package can't find my data files
1. Check that your environment variable is set: `echo $ASTRO_DATA_DIR`
2. Verify the directory structure matches the expected layout
3. Ensure the package has read/write permissions for the directory

### Downloads are failing
1. Check that `R21_URL` is set correctly
2. Verify you have internet connectivity
3. Check that the destination directory has write permissions

### Multiple users/systems
Consider using different environment variables on different systems:
- Workstation: `ASTRO_DATA_DIR="/data/astronomy"`
- Laptop: `ASTRO_DATA_DIR="$HOME/astronomy_data"`
- HPC cluster: `ASTRO_DATA_DIR="$SCRATCH/astronomy"`

## Example Setup Script

Save this as `setup_astro_env.sh`:

```bash
#!/bin/bash
# Configuration for astro_utils

# Base data directory (customize this)
export ASTRO_DATA_DIR="/path/to/your/data"

# R21 data URL (optional, will prompt if not set)
export R21_URL="https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/"

# Optional: Override specific directories if needed
# export R21_SPECTRA_DIR="/custom/path/to/spectra"
# export R21_CATALOG_DIR="/custom/path/to/catalogs"
# export SOURCE_SPECTRA_DIR="/custom/path/to/source_spectra"

echo "astro_utils environment configured:"
echo "  ASTRO_DATA_DIR = $ASTRO_DATA_DIR"
echo "  R21_URL = $R21_URL"
```

Source it before running your scripts:
```bash
source setup_astro_env.sh
python your_analysis_script.py
```

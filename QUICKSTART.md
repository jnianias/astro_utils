# Quick Reference: Environment Setup

## Quick Start (30 seconds)

### Option 1: Use Default Location
```bash
# No setup needed! Data will be stored in ~/.astro_data
pip install -e .
```

### Option 2: Custom Location
```bash
# One-time setup
echo 'export ASTRO_DATA_DIR="/your/data/path"' >> ~/.bashrc
echo 'export R21_URL="<your_R21_data_url>"' >> ~/.bashrc
source ~/.bashrc

# Install
pip install -e .
```

### Option 3: Per-Session Setup
```bash
# Run before each session
export ASTRO_DATA_DIR="/your/data/path"
export R21_URL="<your_R21_data_url>"
```

## Environment Variables

| Variable | What It Does | Example |
|----------|--------------|---------|
| `ASTRO_DATA_DIR` | Where to store all data | `/home/user/astronomy` |
| `R21_URL` | Where to download R21 data | (see above) |
| `R21_SPECTRA_DIR` | Override spectra location | `/data/spectra` |
| `R21_CATALOG_DIR` | Override catalog location | `/data/catalogs` |
| `SOURCE_SPECTRA_DIR` | Override source spectra | `/data/sources` |

## Common Scenarios

### Scenario 1: First Time User
```bash
# Let it use defaults
pip install -e .
python your_script.py
# Data will be in ~/.astro_data
```

### Scenario 2: Migrating from Old Code
```bash
# Keep using your existing data location
export ASTRO_DATA_DIR="/media/james/63C4C5633F1EAE9F/phd/lya_outflows"
```

### Scenario 3: Shared Server
```bash
# Each user gets their own space
export ASTRO_DATA_DIR="$HOME/astro_data"
```

### Scenario 4: Using .env File
```bash
# One-time setup
cp .env.example .env
nano .env  # Edit with your paths

# In your Python code
from dotenv import load_dotenv
load_dotenv()
```

## Check Your Setup

```python
from astro_utils import spectroscopy as spectro

# Where will data be stored?
print(spectro.get_data_dir())

# Check specific directories
print(spectro.get_spectra_dir())
print(spectro.get_source_spectra_dir())
```

## Expected Directory Structure

```
$ASTRO_DATA_DIR/
├── muse_catalogs/
│   ├── catalogs/       # FITS catalogs
│   └── spectra/        # Spectra by cluster
│       ├── A2744/
│       ├── MACS0416/
│       └── ...
└── source_spectra/     # Aperture spectra
    ├── A2744/
    └── ...
```

## Troubleshooting

### "Can't find data files"
```bash
# Check your environment
echo $ASTRO_DATA_DIR
# Make sure directory exists
ls -la $ASTRO_DATA_DIR
```

### "Permission denied"
```bash
# Check directory permissions
ls -ld $ASTRO_DATA_DIR
# Fix if needed
chmod 755 $ASTRO_DATA_DIR
```

### "Download failing"
```bash
# Check URL is set
echo $R21_URL
# Test connectivity
curl -I $R21_URL
```

## More Information

- **Detailed config**: See `CONFIG.md`
- **Full documentation**: See `README.md`

## Example Session

```bash
# 1. Set environment
export ASTRO_DATA_DIR="$HOME/my_astro_data"

# 2. Run Python
python3
>>> from astro_utils import spectroscopy as spectro
>>> spec = spectro.load_r21_spec('A2744', '1234', 'E', 'weight_skysub')
>>> # First time: downloads file to $HOME/my_astro_data/muse_catalogs/spectra/A2744/
>>> # Next time: uses cached file
```
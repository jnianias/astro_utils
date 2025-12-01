# New Spectral Stacking Functions

## Summary
Added two new functions to `astro_utils` for spectral analysis:

### 1. `astro_utils.spectroscopy.stack_spectra_across_sources()`
**Location**: `astro_utils/spectroscopy.py`

**Purpose**: Stack spectral lines across multiple sources onto a common velocity grid.

**Key Features**:
- Handles velocity frame alignment (systemic or Lyman-α)
- Multiple weighting schemes:
  - `'inverse_variance'`: Optimal for Gaussian errors (default)
  - `'uniform'`: Robust to outliers
  - `'inverse_error'`: Intermediate option
- Built on top of `avg_lines()` for within-source line averaging
- Returns stacked spectrum + error + number of sources

**Example Usage**:
```python
from astro_utils import spectroscopy as auspec
import numpy as np

# Stack low-ionization absorption lines
vel, flux, err, n = auspec.stack_spectra_across_sources(
    table=catalog,
    lines=['SiII1260', 'CII1334'],
    velocity_frame='systemic',
    weighting='inverse_variance',
    absorption=True,
    mask=catalog['DV_LI_ABS'] > -np.inf
)
print(f"Stacked {n} sources")
```

### 2. `astro_utils.fitting.flatten_spectrum()`
**Location**: `astro_utils/fitting.py`

**Purpose**: Remove linear continuum trends from spectra.

**Key Features**:
- Fits and removes linear slope
- Preserves continuum level
- Useful for absorption line normalization
- Optionally returns the fitted continuum model

**Example Usage**:
```python
from astro_utils import fitting as aufit

# Remove linear trend
flat_spec = aufit.flatten_spectrum(spectrum)

# Or get the continuum model too
flat_spec, continuum = aufit.flatten_spectrum(spectrum, return_continuum=True)
```

## Migration Guide for S09 Notebook

### Before:
```python
# Manual stacking with lots of code
speclists_abs = {'SINGLE': [[], []], 'DOUBLE': [[], []]}
for row in megatab:
    vel, spec, specerr = avg_lines(row, lines, ...)
    vel += row['DELTAV_LYA']
    spec = np.interp(newvel, vel, spec)
    speclists_abs['SINGLE'][0].append(spec)
    speclists_abs['SINGLE'][1].append(specerr)

# Manual weighted averaging
avgweights = np.nanmedian(np.array(speclists_abs['SINGLE'][1]), axis=1) ** (-2)
avgspec = np.nansum([x * e for x, e in zip(speclists_abs['SINGLE'][0], avgweights)], axis=0) / np.nansum(avgweights)
```

### After:
```python
# Clean, one-line stacking
vel, flux, err, n = auspec.stack_spectra_across_sources(
    table=megatab,
    lines=['SiII1260', 'CII1334'],
    velocity_frame='systemic',
    weighting='inverse_variance',
    mask=(megatab['SNRB'] <= 3.0) & (megatab['Z'] < 4.0)
)
```

## Benefits
1. **Cleaner notebooks**: Complex stacking logic encapsulated in reusable functions
2. **Error propagation**: Proper uncertainty handling built-in
3. **Flexibility**: Easy to switch between weighting schemes and velocity frames
4. **Testability**: Functions can be unit tested independently
5. **Documentation**: Comprehensive docstrings with examples

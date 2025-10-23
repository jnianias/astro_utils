#!/usr/bin/env python3
"""
Test the check_multiple_peaks function to detect suspicious spectra
with multiple comparable emission features.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astro_utils.fitting import check_multiple_peaks

def gaussian(x, amp, center, sigma):
    """Simple Gaussian function."""
    return amp * np.exp(-0.5 * ((x - center) / sigma)**2)

print("=" * 70)
print("TEST: check_multiple_peaks - Detecting suspicious spectra")
print("=" * 70)

# Create wavelength array
wave = np.linspace(5000, 5100, 1000)
err = np.ones_like(wave) * 0.01

# Test Case 1: Clean single line
print("\n" + "=" * 70)
print("TEST CASE 1: Single clean emission line")
print("=" * 70)

# Fitted line parameters
fitted_wave = 5050.0
fitted_amp = 1.0
fitted_width = 2.5  # FWHM

# Create model (what we fitted)
model = gaussian(wave, fitted_amp, fitted_wave, fitted_width / 2.355)

# Add noise
np.random.seed(42)
noise = np.random.normal(0, err)
observed = model + noise

# Residuals (what we check)
residuals = observed - model

result = check_multiple_peaks(wave, residuals, err, 
                              fitted_wave, fitted_amp, fitted_width,
                              n_fit_params=3)

print(f"Reduced χ²: {result['reduced_chi2']:.2f}")
print(f"Suspicious: {result['suspicious']}")
print(f"Number of comparable peaks: {result['n_comparable_peaks']}")
print(f"Flag: '{result['flag']}'")
print(f"Message: {result['message']}")

# Test Case 2: Two similar emission lines
print("\n" + "=" * 70)
print("TEST CASE 2: Two emission lines of similar strength")
print("=" * 70)

# Fitted line (weaker one)
fitted_wave = 5040.0
fitted_amp = 0.8
fitted_width = 2.5

model_fitted = gaussian(wave, fitted_amp, fitted_wave, fitted_width / 2.355)

# But there's another similar line we didn't fit
other_line = gaussian(wave, 0.9, 5060.0, 3.0 / 2.355)

observed = model_fitted + other_line + noise
residuals = observed - model_fitted

result = check_multiple_peaks(wave, residuals, err,
                              fitted_wave, fitted_amp, fitted_width,
                              n_fit_params=3)

print(f"Reduced χ²: {result['reduced_chi2']:.2f}")
print(f"Suspicious: {result['suspicious']}")
print(f"Number of comparable peaks: {result['n_comparable_peaks']}")
print(f"Flag: '{result['flag']}'")
print(f"Message: {result['message']}")

if result['suspicious']:
    for i, peak in enumerate(result['peak_info']):
        print(f"\n  Peak {i+1}:")
        print(f"    Wavelength: {peak['wavelength']:.2f} Å")
        print(f"    SNR: {peak['snr']:.1f}")
        print(f"    Amplitude ratio: {peak['amplitude_ratio']:.2f}")
        print(f"    Width ratio: {peak['width_ratio']:.2f}")
        print(f"    Separation: {peak['separation']:.2f} Å ({peak['separation']/fitted_width:.1f} × FWHM)")

# Test Case 3: Multiple weak peaks
print("\n" + "=" * 70)
print("TEST CASE 3: One strong line + multiple weak features")
print("=" * 70)

fitted_wave = 5050.0
fitted_amp = 2.0
fitted_width = 2.5

model_fitted = gaussian(wave, fitted_amp, fitted_wave, fitted_width / 2.355)

# Add several weak features
weak_features = (
    gaussian(wave, 0.3, 5030.0, 2.0 / 2.355) +
    gaussian(wave, 0.4, 5070.0, 2.5 / 2.355)
)

observed = model_fitted + weak_features + noise
residuals = observed - model_fitted

result = check_multiple_peaks(wave, residuals, err,
                              fitted_wave, fitted_amp, fitted_width,
                              n_fit_params=3)

print(f"Reduced χ²: {result['reduced_chi2']:.2f}")
print(f"Suspicious: {result['suspicious']}")
print(f"Number of comparable peaks: {result['n_comparable_peaks']}")
print(f"Flag: '{result['flag']}'")
print(f"Message: {result['message']}")
print("(Should NOT be flagged - weak features < 50% of fitted peak)")

# Test Case 4: One strong + one similar strength
print("\n" + "=" * 70)
print("TEST CASE 4: Two lines - one fitted, one 70% as strong")
print("=" * 70)

fitted_wave = 5045.0
fitted_amp = 1.5
fitted_width = 3.0

model_fitted = gaussian(wave, fitted_amp, fitted_wave, fitted_width / 2.355)

# Another line at 70% strength
other_strong = gaussian(wave, 1.05, 5070.0, 3.5 / 2.355)

observed = model_fitted + other_strong + noise
residuals = observed - model_fitted

result = check_multiple_peaks(wave, residuals, err,
                              fitted_wave, fitted_amp, fitted_width,
                              n_fit_params=3)

print(f"Reduced χ²: {result['reduced_chi2']:.2f}")
print(f"Suspicious: {result['suspicious']}")
print(f"Number of comparable peaks: {result['n_comparable_peaks']}")
print(f"Flag: '{result['flag']}'")
print(f"Message: {result['message']}")

if result['suspicious']:
    for i, peak in enumerate(result['peak_info']):
        print(f"\n  Peak {i+1}:")
        print(f"    Wavelength: {peak['wavelength']:.2f} Å")
        print(f"    SNR: {peak['snr']:.1f}")
        print(f"    Amplitude ratio: {peak['amplitude_ratio']:.2f}")
        print(f"    Width ratio: {peak['width_ratio']:.2f}")

print("\n" + "=" * 70)
print("Summary:")
print("- Clean single line: Should NOT flag")
print("- Two similar lines: SHOULD flag with 'm'")
print("- Weak features: Should NOT flag (below threshold)")
print("- 70% strength companion: SHOULD flag (above 50% threshold)")
print("=" * 70)

#!/usr/bin/env python3
"""
Test autocorrelation for spectroscopy use case:
Short correlations (1-5 pixels) relevant for spurious line detection.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astro_utils.fitting import gen_corr_noise, autocorr_length

print("=" * 70)
print("SPECTROSCOPY USE CASE: Short-scale correlation estimation")
print("=" * 70)

# Test short correlation lengths relevant for spurious line detection
test_taus = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
n_points = 1000
max_lag = 10  # Typical for spectroscopy

print(f"\nGenerating {n_points} point series with uniform errors")
print(f"Using max_lag = {max_lag} (typical for line detection)")
print(f"Testing τ values: {test_taus}")
print("\n" + "=" * 70)

results = []

for true_tau in test_taus:
    print(f"\n{'=' * 70}")
    print(f"TRUE τ = {true_tau:.1f} pixels")
    print(f"{'=' * 70}")
    
    # Generate correlated noise
    yerr = np.ones(n_points)
    noise = gen_corr_noise(yerr, true_tau)
    
    # Estimate tau (as would be done from fit residuals)
    wave = np.arange(n_points)
    estimated_tau, inflation_factor = autocorr_length(wave, noise, yerr, max_lag=max_lag, baseline_order=None)
    
    # Calculate error
    error = estimated_tau - true_tau
    relative_error_pct = 100 * error / true_tau
    
    print(f"\n>>> RESULT:")
    print(f"    True τ       = {true_tau:.2f}")
    print(f"    Estimated τ  = {estimated_tau:.2f}")
    print(f"    Error        = {error:+.2f} ({relative_error_pct:+.1f}%)")
    
    results.append({
        'true_tau': true_tau,
        'estimated_tau': estimated_tau,
        'error': error,
        'relative_error_pct': relative_error_pct
    })

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Short-scale correlations (spectroscopy)")
print("=" * 70)
print(f"{'True τ':<10} {'Est. τ':<10} {'Error':<10} {'Rel. Error':<12}")
print("-" * 70)

for r in results:
    print(f"{r['true_tau']:<10.1f} {r['estimated_tau']:<10.2f} "
          f"{r['error']:<+10.2f} {r['relative_error_pct']:<+12.1f}%")

errors = [r['error'] for r in results]
rel_errors = [r['relative_error_pct'] for r in results]

print("-" * 70)
print(f"Mean absolute error: {np.mean(np.abs(errors)):.2f} pixels")
print(f"RMS error:           {np.sqrt(np.mean(np.array(errors)**2)):.2f} pixels")
print(f"Mean relative error: {np.mean(rel_errors):+.1f}%")
print(f"Mean |rel. error|:   {np.mean(np.abs(rel_errors)):.1f}%")

# For spectroscopy, most important: small τ values (1-3 pixels)
short_scale = [r for r in results if r['true_tau'] <= 3.0]
short_errors = [abs(r['relative_error_pct']) for r in short_scale]
print(f"\nFor τ ≤ 3 pixels (most critical for spurious lines):")
print(f"  Mean |rel. error|: {np.mean(short_errors):.1f}%")

all_good = all(abs(r['relative_error_pct']) < 30 for r in results)
print("\n" + "=" * 70)
if all_good:
    print("✓ SUCCESS: All estimates within 30% of true value")
else:
    print("✗ WARNING: Some estimates differ by >30%")
print("=" * 70)

# Conservative assessment
conservative_ok = all(r['estimated_tau'] >= 0.7 * r['true_tau'] for r in results)
if conservative_ok:
    print("✓ CONSERVATIVE: No severe underestimation (all > 70% of true τ)")
else:
    print("✗ WARNING: Some severe underestimation detected")
print("=" * 70)

#!/usr/bin/env python3
"""
Self-consistency test: Generate correlated noise with known tau,
then recover it with autocorr_length estimation.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import astro_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astro_utils.fitting import gen_corr_noise, autocorr_length

print("=" * 70)
print("SELF-CONSISTENCY TEST: Generate noise → Estimate τ → Compare")
print("=" * 70)

# Test different correlation lengths
test_taus = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
n_points = 2000  # Use long series for good statistics

print(f"\nGenerating {n_points} point series with uniform errors")
print(f"Testing τ values: {test_taus}")
print("\n" + "=" * 70)

results = []

for true_tau in test_taus:
    print(f"\n{'=' * 70}")
    print(f"TRUE τ = {true_tau:.1f} pixels")
    print(f"{'=' * 70}")
    
    # Generate correlated noise with known tau
    yerr = np.ones(n_points)
    noise = gen_corr_noise(yerr, true_tau)
    
    # Try to estimate tau back
    wave = np.arange(n_points)
    max_lag_to_use = max(20, int(3 * true_tau))
    
    print(f"\nUsing max_lag = {max_lag_to_use}")
    estimated_tau, inflation_factor = autocorr_length(wave, noise, yerr, max_lag=max_lag_to_use, baseline_order=None)
    
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
print("SUMMARY")
print("=" * 70)
print(f"{'True τ':<10} {'Est. τ':<10} {'Error':<10} {'Rel. Error':<12}")
print("-" * 70)

for r in results:
    print(f"{r['true_tau']:<10.1f} {r['estimated_tau']:<10.2f} "
          f"{r['error']:<+10.2f} {r['relative_error_pct']:<+12.1f}%")

# Statistics
errors = [r['error'] for r in results]
rel_errors = [r['relative_error_pct'] for r in results]

print("-" * 70)
print(f"Mean absolute error: {np.mean(np.abs(errors)):.2f} pixels")
print(f"RMS error:           {np.sqrt(np.mean(np.array(errors)**2)):.2f} pixels")
print(f"Mean relative error: {np.mean(rel_errors):+.1f}%")
print(f"Mean |rel. error|:   {np.mean(np.abs(rel_errors)):.1f}%")

# Check if all estimates are within 30% of true value
all_good = all(abs(r['relative_error_pct']) < 30 for r in results)
print("\n" + "=" * 70)
if all_good:
    print("✓ SUCCESS: All estimates within 30% of true value")
else:
    print("✗ WARNING: Some estimates differ by >30%")
print("=" * 70)

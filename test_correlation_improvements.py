#!/usr/bin/env python3
"""
Test script to verify the improved autocorrelation length estimation.
This demonstrates:
1. The exponential fit approach vs threshold crossing
2. The consistency between gen_corr_noise and autocorr_length
"""

import numpy as np
import matplotlib.pyplot as plt
from astro_utils.fitting import gen_corr_noise, autocorr_length

# Test 1: Generate noise with known correlation length and recover it
print("=" * 60)
print("TEST 1: Generate correlated noise and recover τ")
print("=" * 60)

for true_tau in [2.0, 5.0, 10.0]:
    print(f"\n>>> True τ = {true_tau}")
    
    # Generate long correlated noise sequence
    n_points = 1000
    yerr = np.ones(n_points)
    noise = gen_corr_noise(yerr, true_tau)
    
    # Estimate correlation length
    wave = np.arange(n_points)
    estimated_tau, inflation_factor = autocorr_length(wave, noise, yerr, max_lag=int(3 * true_tau))
    
    print(f"Estimated τ = {estimated_tau:.2f}")
    print(f"Relative error: {100 * (estimated_tau - true_tau) / true_tau:.1f}%")

# Test 2: Verify ACF of generated noise matches theory
print("\n" + "=" * 60)
print("TEST 2: Verify ACF(k) = exp(-k/τ) for generated noise")
print("=" * 60)

true_tau = 5.0
n_points = 5000
yerr = np.ones(n_points)
noise = gen_corr_noise(yerr, true_tau)

# Calculate ACF manually
noise_centered = noise - np.mean(noise)
acf = np.correlate(noise_centered, noise_centered, mode='full')[len(noise_centered)-1:]
acf = acf / acf[0]

# Compare to theory
lags = np.arange(0, 20)
acf_theory = np.exp(-lags / true_tau)
acf_measured = acf[:20]

print(f"\nTrue τ = {true_tau}")
print(f"{'Lag':<5} {'Theory':<10} {'Measured':<10} {'Diff':<10}")
print("-" * 40)
for k in range(0, 20, 2):
    diff = acf_measured[k] - acf_theory[k]
    print(f"{k:<5} {acf_theory[k]:<10.4f} {acf_measured[k]:<10.4f} {diff:<10.4f}")

# Test 3: Check the phi formula
print("\n" + "=" * 60)
print("TEST 3: Verify phi = exp(-1/τ) gives ACF(τ) ≈ 1/e")
print("=" * 60)

for tau in [2.0, 5.0, 10.0]:
    phi = np.exp(-1.0 / tau)
    acf_at_tau = phi ** tau
    expected = np.exp(-1)  # 1/e ≈ 0.368
    print(f"τ={tau:4.1f}: phi={phi:.4f}, ACF(τ)={acf_at_tau:.4f}, expected={expected:.4f}, diff={acf_at_tau-expected:.4f}")

print("\n" + "=" * 60)
print("All tests complete!")
print("=" * 60)

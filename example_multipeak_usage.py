#!/usr/bin/env python3
"""
Example of how the user would access the multipeak flag from refit_other_line
"""

import numpy as np

# Simulated example of calling refit_other_line
# (In reality, this would be called with real data)

print("=" * 70)
print("Example: Using multipeak_flag from refit_other_line")
print("=" * 70)

# When user calls refit_other_line, they now get 5 return values:
# param_dict, error_dict, model, reduced_chisq, multipeak_flag = refit_other_line(...)

# Example 1: Clean fit
print("\nExample 1: Clean single-line fit")
print("-" * 70)
print("params, errors, model, chi2, flag = refit_other_line(...)")
print("\nResults:")
print("  reduced_chi2: 1.2")
print("  multipeak_flag: ''")
print("  → GOOD: No flag, fit is clean")

# Example 2: Multiple peaks detected
print("\nExample 2: Multiple comparable peaks detected")
print("-" * 70)
print("params, errors, model, chi2, flag = refit_other_line(...)")
print("\nResults:")
print("  reduced_chi2: 45.3")
print("  multipeak_flag: 'm'")
print("  → WARNING: Multiple peaks found, fit may be unreliable")
print("  → User should inspect spectrum manually")

# Example 3: Poor fit but no comparable peaks
print("\nExample 3: Poor fit, but no comparable peaks")
print("-" * 70)
print("params, errors, model, chi2, flag = refit_other_line(...)")
print("\nResults:")
print("  reduced_chi2: 8.5")
print("  multipeak_flag: ''")
print("  → Poor fit, but peaks have different amplitude/width")
print("  → Might be baseline issue or absorption feature")

print("\n" + "=" * 70)
print("Usage pattern:")
print("=" * 70)
print("""
# Basic usage:
params, errors, model, chi2, flag = refit_other_line(
    wave, spec, spec_err, row, 
    line_name='CIV1548',
    bootstrap_params={'autocorrelation': True}
)

# Check the flag:
if flag == 'm':
    print("WARNING: Multiple comparable peaks detected!")
    print("This fit may be contaminated by other emission lines")
    # User can choose to:
    # - Reject this fit
    # - Mark it for manual inspection
    # - Try fitting multiple lines simultaneously
    
# Access full diagnostic info from fit_line:
# (if calling fit_line directly instead of refit_other_line)
fit_result = fit_line(wave, spec, spec_err, 'CIV1548', initial_guesses, bounds)
if 'peak_check' in fit_result:
    peak_info = fit_result['peak_check']
    print(f"Chi-square: {peak_info['reduced_chi2']:.2f}")
    print(f"Message: {peak_info['message']}")
    if peak_info['suspicious']:
        for peak in peak_info['peak_info']:
            print(f"  Peak at {peak['wavelength']:.2f} Å")
            print(f"    Amplitude ratio: {peak['amplitude_ratio']:.2f}")
            print(f"    SNR: {peak['snr']:.1f}")
""")

print("=" * 70)

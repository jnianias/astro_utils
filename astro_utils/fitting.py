
from . import constants as const
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation, norm
import warnings
from . import models as mdl
from astropy.stats import sigma_clipped_stats
import traceback
from error_propagation import Complex
from . import spectroscopy as spectro
import matplotlib.pyplot as plt
import error_propagation as ep

# Import moved components for backwards compatibility
from .spectroscopy import generate_spec_mask, mask_skylines, mask_otherlines

# Import dictionary of line wavelengths and speed of light
wavedict    = const.wavedict
doubletdict = const.doublets
skylinedict = const.skylines
c           = const.c  # speed of light in km/s


def which_fit_method(linename):
    """
    Determines the fitting method based on the line name.
    If you provide the principal line of a doublet, it will return 'doublet',
    but if you use a secondary line, it will warn you and return 'single'.

    linename: name of the line (needs to be in wavedict)
    """
    doublet_principals  = list(doubletdict.keys())
    doublet_secondaries = [doubletdict[key][1] for key in doublet_principals 
                           if isinstance(doubletdict[key], tuple)]
    
    if linename in doublet_principals:
        return 'doublet'
    elif linename not in doublet_secondaries:
        return 'single'
    else:
        warnings.warn(f"Line {linename} is a secondary doublet line; consider skipping!")
        return 'single'
    

from numpy.polynomial import Polynomial as nppoly

def autocorr_length(wave, spec, yerr, max_lag=10, baseline_order=None):
    """Estimate noise correlation length from spectral residuals.
    
    Designed for short astronomical spectra (30-100 pixels): estimates short-scale
    correlations (typically 1-5 pixels) that could cause correlated noise peaks to
    mimic emission lines.
    
    Returns the e-folding length τ where ACF(τ) = 1/e ≈ 0.368.
    
    Algorithm:
    ----------
    Optimized for short, noisy spectra where traditional ACF tail estimation is
    unreliable. Uses a simplified two-stage approach:
    
    **Stage 1: Significance Test**
        Tests if ACF(1) exceeds 3σ threshold based on statistical uncertainty
        (~3/sqrt(n)). If not, returns τ=1 (no significant correlation).
        Prevents estimating correlation from noise fluctuations.
    
    **Stage 2: Exponential Fit & Boundary Detection**
        Fits exponential decay to first few lags (most reliable estimates).
        Uses fixed threshold (0.05) for boundary detection - robust for
        short spectra where ACF tail is too noisy for reliable estimation.
        
    Uses unbiased ACF estimator: divides by actual number of pairs at each lag
    to avoid systematic negative bias at large lags.
    
    Philosophy: Simple and robust for short spectra. Conservative for detection
    (require clear lag-1 correlation), but uses fixed thresholds to avoid
    unreliable noise floor estimation from noisy ACF tails.
    
    Parameters:
    -----------
    wave : array
        Wavelength array (used for baseline fitting)
    spec : array  
        Spectral residuals (observed - model) or flux values
    yerr : array
        Error estimates for each pixel (for χ²_red calculation)
    max_lag : int
        Maximum lag to search for correlations (default: 10)
        For line detection, use ~5-10 pixels
    baseline_order : int or None
        Polynomial order for baseline removal before ACF
        Use None (mean removal) for residuals, 0-2 for raw spectra
    
    Returns:
    --------
    tau : float
        Correlation length in pixels (>= 1.0)
        Returns 1.0 if no significant correlation detected at lag 1
    inflation_factor : float
        Error inflation factor based on χ²_red (>= 1.0)
        Accounts for systematic excess variance beyond formal errors
    
    Notes:
    ------
    For spectroscopic data, correlation at lag 1 (adjacent pixels) is expected
    if any correlation exists. Higher lags without lag-1 correlation are 
    physically unrealistic, so the lag-1 test is sufficient for detection.
    """
    # 1. Baseline removal with diagnostics
    if baseline_order is not None:
        p = nppoly.fit(wave, spec, baseline_order)
        residuals = spec - p(wave)
        print(f"Baseline: polynomial order {baseline_order}")
    else:
        residuals = spec - np.mean(spec)
        print(f"Baseline: mean removal only")

    # Check for systematic excess variance (Ly-α forest, sky lines, etc.)
    # Use robust estimator: median of |residuals|/yerr
    # This is less sensitive to outliers than χ²_red
    normalized_residuals = np.abs(residuals) / yerr
    median_normalized_residual = np.median(normalized_residuals)
    
    # Trigger conservatively using median to avoid outlier sensitivity
    # But correct using proper Gaussian calibration for accurate scaling
    threshold = 1.0
    if median_normalized_residual > threshold:
        # For Gaussian noise, median(|z|) ≈ 0.675, so scale to get true σ
        inflation_factor = median_normalized_residual / 0.675
        print(f"Systematic excess variance detected: inflating errors by {inflation_factor:.2f}x")
    else:
        inflation_factor = 1.0

    # 2. ACF calculation with unbiased estimator
    n = len(residuals)
    # Remove mean to ensure zero-mean signal
    residuals = residuals - np.mean(residuals)
    
    # Compute ACF using unbiased estimator
    # For each lag k, divide the sum by the actual number of pairs (n-k)
    max_acf_lag = n  # Can compute ACF up to lag n-1
    acf = np.zeros(max_acf_lag)
    
    for k in range(max_acf_lag):
        if n - k > 0:
            # Sum of products for lag k, divided by number of pairs
            acf[k] = np.sum(residuals[:n-k] * residuals[k:]) / (n - k)
        else:
            break
    
    if acf[0] <= 0:
        print("WARNING: Non-positive ACF(0); returning τ=1")
        return 1.0, inflation_factor  # Protection against bad normalization
    
    # Normalize so ACF(0) = 1
    acf = acf / acf[0]

    # 3. Simplified robust approach for short spectra
    # Don't try to estimate noise floor from unreliable tail - just use lag-1 correlation
    
    # Stage 1: Check if there's significant correlation at lag 1
    # Use 2σ threshold - more lenient to catch real correlations in noisy data
    # For short spectra (n < 100), ACF[1] standard error ~ 1/sqrt(n)
    acf_1_stderr = 1.0 / np.sqrt(n)
    detection_threshold = 2.0 * acf_1_stderr  # 2-sigma detection (more sensitive)
    
    print(f"\nACF[1] = {acf[1]:.3f} ± {acf_1_stderr:.3f}")
    print(f"Detection threshold (2σ): {detection_threshold:.3f}")
    
    if acf[1] < detection_threshold:
        print(f"No significant correlation detected at lag 1")
        print("Returning τ=1 (uncorrelated)")
        return 1.0, inflation_factor
    
    # Stage 2: Estimate τ from early lags using exponential fit
    # Use first few lags (3-7) where ACF is most reliable
    print(f"Correlation detected: ACF[1]={acf[1]:.3f} > {detection_threshold:.3f}")
    
    # Fit exponential through lags 1-5 (or fewer if spectrum is very short)
    max_fit_lag = min(7, max_lag, n // 4)  # Don't use more than 25% of data
    
    # Use fixed threshold for boundary detection (robust for short, noisy spectra)
    boundary_threshold = 0.05
    
    # 5. Find where ACF drops below boundary threshold (fallback method)
    crossing_lag = None
    for k in range(1, min(max_lag, len(acf)-1)):
        if acf[k] < boundary_threshold:
            crossing_lag = k
            break
    
    # 6. Fit exponential decay ACF(k) = exp(-k/τ) to estimate τ
    # For spectroscopy: focus on short lags (1-5 pixels) where noise mimics lines
    # Don't fit too far - long-range correlations from baseline structure aren't relevant
    
    # Restrict fitting to short lags relevant for line detection
    # Use crossing lag as guide, but cap at max_lag
    if crossing_lag is not None and 3 <= crossing_lag <= max_lag:
        # Found a clear crossing within search range
        fit_up_to = crossing_lag
    elif crossing_lag is not None and crossing_lag < 3:
        # Very short correlation - fit up to lag 5 to be sure
        fit_up_to = min(5, max_lag)
    else:
        # No crossing found - use max_lag but warn
        fit_up_to = max_lag
        if acf[max_lag] > boundary_threshold:
            print(f"WARNING: ACF({max_lag}) = {acf[max_lag]:.3f} still above boundary threshold")
            print(f"         Long-range correlation detected - may be baseline structure")
    
    # Fit exponential to early lags
    lags = np.arange(1, min(fit_up_to + 1, len(acf)))
    acf_to_fit = acf[1:min(fit_up_to + 1, len(acf))]
    
    # Only fit where ACF is positive and meaningful (> 0.05)
    min_acf_value = 0.05
    valid_mask = acf_to_fit > min_acf_value
    n_valid = np.sum(valid_mask)
    
    print(f"Fitting range: lags 1-{len(lags)}, {n_valid}/{len(lags)} valid points (ACF > {min_acf_value})")
    
    if n_valid >= 3:
        try:
            # Constrained exponential fit: ACF(k) = exp(-k/τ)
            # This enforces ACF(0) = 1 as required for autocorrelation
            # We fit: log(ACF) = -k/τ, forcing intercept = 0
            
            # Weighted least squares for log(ACF) vs lag (zero intercept)
            weights = 1.0 / np.sqrt(lags[valid_mask])
            
            # Manual weighted fit with zero intercept: slope = Σ(w*x*y) / Σ(w*x²)
            x = lags[valid_mask]
            y = np.log(acf_to_fit[valid_mask])
            w = weights**2
            slope = np.sum(w * x * y) / np.sum(w * x * x)
            tau_fit = -1.0 / slope
            
            # Sanity check: expect short correlations (0.5 to ~2*max_lag)
            # For spurious line detection, τ > max_lag means long-range structure
            if 0.5 <= tau_fit <= 2 * max_lag:
                print(f"Fitted τ = {tau_fit:.2f} pixels (exponential fit)")
                if crossing_lag is not None:
                    print(f"(Threshold crossing at lag {crossing_lag})")
                return max(1.0, tau_fit), inflation_factor
            else:
                print(f"Fitted τ = {tau_fit:.2f} outside expected range [0.5, {2*max_lag}]")
                # If too large, cap at max_lag (conservative)
                if tau_fit > 2 * max_lag:
                    print(f"Using τ = {max_lag} (capped - likely baseline structure)")
                    return float(max_lag), inflation_factor
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
            print(f"Exponential fit failed: {e}")
    else:
        print(f"Not enough valid points for fitting (need ≥3, have {n_valid})")
        # Fallback: estimate τ from ACF[1] directly
        # For exponential decay: ACF(1) = exp(-1/τ) → τ = -1/log(ACF[1])
        if acf[1] > 0.1:  # Reasonable correlation at lag 1
            tau_simple = -1.0 / np.log(acf[1])
            print(f"Fallback: estimating τ from ACF[1]={acf[1]:.3f} → τ={tau_simple:.2f}")
            return max(1.0, tau_simple), inflation_factor

    # 6. Final fallback to threshold crossing or max_lag
    if crossing_lag is not None:
        print(f"Using threshold crossing at lag {crossing_lag} (ACF={acf[crossing_lag]:.3f})")
        return float(crossing_lag), inflation_factor
    else:
        print(f"WARNING: No crossing found within max_lag={max_lag} - returning max_lag")
        if acf[min(max_lag, len(acf)-1)] > boundary_threshold:
            print(f"WARNING: ACF({max_lag}) = {acf[min(max_lag, len(acf)-1)]:.3f} still above boundary threshold")
        return float(max_lag), inflation_factor

def gen_corr_noise(yerr, corr_len, size=None):
    """Generate correlated noise using AR(1) process.
    
    For an AR(1) process with parameter phi, the ACF is:
        ACF(k) = phi^k
    
    We want ACF(τ) = exp(-1) where τ is the correlation length (e-folding scale).
    Therefore: phi^τ = exp(-1)  =>  phi = exp(-1/τ)
    
    Args:
        yerr: Array of error values
        corr_len: Correlation length τ in pixels (e-folding scale)
        size: Optional output size (defaults to len(yerr))
    Returns:
        Correlated noise array with same shape as yerr/size
    """
    if corr_len <= 1:
        return np.random.normal(scale=yerr, size=size)

    n = len(yerr) if size is None else size

    # AR(1) process: noise[i] = phi * noise[i-1] + sqrt(1-phi^2) * epsilon[i]
    noise = np.zeros(n)
    noise[0] = np.random.normal()

    # For correlation length τ (e-folding scale): phi = exp(-1/τ)
    phi = np.exp(-1.0 / corr_len)

    for i in range(1, n):
        noise[i] = phi * noise[i-1] + np.sqrt(1-phi**2) * np.random.normal()

    # Scale by errors while preserving correlation structure
    return noise * (yerr[:n] if yerr.ndim > 0 else yerr) 


def check_multiple_peaks(wave, residuals, err, fitted_peak_wave, fitted_amplitude, 
                         fitted_width, n_fit_params=3, min_separation=3, 
                         amplitude_ratio_threshold=0.5, width_ratio_range=(0.5, 2.0), 
                         detection_threshold=3.0, chi2_threshold=3.0):
    """
    Check if there are other unexplained peaks in the residuals similar to the fitted line.
    
    This detects cases where multiple peaks of comparable strength exist, suggesting
    the fitted line may not be unique or the spectrum is contaminated by complex 
    structure (e.g., multiple emission lines, broad absorption features, etc.).
    
    Only performs peak search if reduced chi-square is high (poor fit), otherwise
    returns immediately with no flag.
    
    Parameters:
    -----------
    wave : array
        Wavelength array
    residuals : array
        Residuals after subtracting the fitted model (observed - model)
    err : array
        Error array (for significance calculation)
    fitted_peak_wave : float
        Wavelength of the fitted peak
    fitted_amplitude : float
        Amplitude (height) of the fitted peak
    fitted_width : float
        FWHM or sigma of the fitted peak (in same units as wave)
    n_fit_params : int
        Number of parameters in the fit (for reduced chi-square calculation)
        (default: 3, e.g., amplitude, center, width)
    min_separation : float
        Minimum separation in units of fitted_width to consider peaks as distinct
        (default: 3, i.e., peaks must be >3*FWHM apart)
    amplitude_ratio_threshold : float
        Flag if other peaks have amplitude > this fraction of fitted peak
        (default: 0.5, i.e., flag if other peaks are >50% as strong)
    width_ratio_range : tuple
        Flag if other peaks have widths within this range of fitted width
        (default: (0.5, 2.0), i.e., half to double the width)
    detection_threshold : float
        Minimum SNR for a peak in residuals to be considered significant
        (default: 3.0 sigma)
    chi2_threshold : float
        Only search for peaks if reduced chi-square > this value
        (default: 3.0, indicating significant unexplained structure)
    
    Returns:
    --------
    dict with keys:
        'suspicious': bool - True if suspicious peaks found AND reduced chi2 > threshold
        'n_comparable_peaks': int - Number of comparable peaks found
        'peak_info': list of dicts - Info about each suspicious peak
            Each dict contains: wavelength, amplitude, snr, width, 
            amplitude_ratio, width_ratio, separation
        'flag': str - Suggested flag ('m' for multiple peaks, or '')
        'reduced_chi2': float - Calculated reduced chi-square of fit
        'message': str - Human-readable explanation of result
    
    Example:
    --------
    >>> result = check_multiple_peaks(wave, residuals, err, 5007.0, 1e-17, 2.5)
    >>> if result['suspicious']:
    >>>     print(f"Found {result['n_comparable_peaks']} comparable peaks")
    """
    from scipy.signal import find_peaks
    
    # First check: Calculate reduced chi-square
    chi2 = np.sum((residuals / err) ** 2)
    n_data = len(residuals)
    dof = n_data - n_fit_params
    reduced_chi2 = chi2 / dof
    
    # If fit is good (low chi-square), no need to search for peaks
    if reduced_chi2 <= chi2_threshold:
        return {
            'suspicious': False,
            'n_comparable_peaks': 0,
            'peak_info': [],
            'flag': '',
            'reduced_chi2': reduced_chi2,
            'message': f'Good fit (χ²_red = {reduced_chi2:.2f} ≤ {chi2_threshold}), no peak search needed'
        }
    
    # Find peaks in the residuals (positive and negative)
    # Look at absolute value to catch both emission and absorption features
    abs_residuals = np.abs(residuals)
    significance = abs_residuals / err
    
    # Find peaks in the significance array
    # Use a minimum height of detection_threshold sigma
    peaks_idx, properties = find_peaks(significance, 
                                       height=detection_threshold,
                                       prominence=detection_threshold * 0.5)
    
    if len(peaks_idx) == 0:
        return {
            'suspicious': False,
            'n_comparable_peaks': 0,
            'peak_info': [],
            'flag': '',
            'reduced_chi2': reduced_chi2,
            'message': f'Poor fit (χ²_red = {reduced_chi2:.2f}) but no significant peaks found'
        }
    
    # Get peak properties
    peak_waves = wave[peaks_idx]
    peak_amplitudes = abs_residuals[peaks_idx]  # Use absolute residuals where peaks were found
    peak_snrs = significance[peaks_idx]
    
    # Estimate widths of peaks using scipy's peak_widths
    from scipy.signal import peak_widths
    widths_idx, width_heights, left_ips, right_ips = peak_widths(
        significance, peaks_idx, rel_height=0.5
    )
    
    # Convert width from indices to wavelength units
    delta_wave = np.median(np.diff(wave))
    peak_widths = widths_idx * delta_wave
    
    # Filter peaks:
    # 1. Exclude the fitted peak itself (within min_separation * fitted_width)
    # 2. Keep only peaks with comparable amplitude
    # 3. Keep only peaks with comparable width
    
    suspicious_peaks = []
    
    for i, (pw, pa, psnr, pwidth) in enumerate(zip(peak_waves, peak_amplitudes, 
                                                     peak_snrs, peak_widths)):
        # Check if this is the fitted peak (too close)
        separation = np.abs(pw - fitted_peak_wave)
        if separation < min_separation * fitted_width:
            continue  # This is probably the fitted line itself
        
        # Check amplitude ratio (pa is already absolute value from abs_residuals)
        amplitude_ratio = pa / np.abs(fitted_amplitude)
        if amplitude_ratio < amplitude_ratio_threshold:
            continue  # Too weak
        
        # Check width ratio
        width_ratio = pwidth / fitted_width
        if width_ratio < width_ratio_range[0] or width_ratio > width_ratio_range[1]:
            continue  # Too narrow or too wide
        
        # This peak is suspicious!
        suspicious_peaks.append({
            'wavelength': pw,
            'amplitude': pa,
            'snr': psnr,
            'width': pwidth,
            'amplitude_ratio': amplitude_ratio,
            'width_ratio': width_ratio,
            'separation': separation
        })
    
    n_suspicious = len(suspicious_peaks)
    flag = 'm' if n_suspicious > 0 else ''
    
    if n_suspicious > 0:
        message = f'Poor fit (χ²_red = {reduced_chi2:.2f}): found {n_suspicious} comparable peak(s)'
    else:
        message = f'Poor fit (χ²_red = {reduced_chi2:.2f}) but no comparable peaks (different amplitude/width)'
    
    return {
        'suspicious': n_suspicious > 0,
        'n_comparable_peaks': n_suspicious,
        'peak_info': suspicious_peaks,
        'flag': flag,
        'reduced_chi2': reduced_chi2,
        'message': message
    }


def sigma_to_percentile(sigma):
    """
    Convert sigma level to percentile for non-Gaussian uncertainties.
    
    Parameters:
    sigma (float): Sigma level (e.g., 1, 2, 3 for 1σ, 2σ, 3σ)
    
    Returns:
    float: Percentile value representing the confidence level
    """
    
    # Calculate the two-tailed percentile
    # For Gaussian: P(|X| < kσ) = erf(k/√2)
    confidence_level = norm.cdf(sigma) - norm.cdf(-sigma)
    
    # Convert to percentile (0-100 scale)
    percentile = confidence_level * 100
    
    return percentile


def avgfunc(poptl, errfunc, sig_clip = 7.0):
        if errfunc == 'stddev':
            _scs = sigma_clipped_stats(np.array(poptl), axis=0, maxiters=1, sigma=sig_clip)
            return [_scs[0], _scs[2]]
        else:
            medpopt = np.nanmedian(poptl, axis=0)
            return [medpopt, np.nanmedian(np.abs(poptl - medpopt), axis=0)]

def fit_mc(f, x, y, yerr, p0, bounds=None, niter=500, errfunc='mad',
           return_sample=False, chisq_thresh=np.inf, sig_clip=7.0,
           autocorrelation=False, max_lag=10, baseline_order=0, max_nfev=5000):
    """Monte Carlo fitting with correlated noise handling for spectroscopy.
    
    Fits a model to data and estimates parameter uncertainties via Monte Carlo
    resampling. Can account for correlated noise in residuals to avoid 
    underestimating errors when noise correlation mimics emission line structure.
    
    Parameters:
    -----------
    f : callable
        Model function to fit: f(x, *params)
    x : array
        Independent variable (e.g., wavelength)
    y : array
        Dependent variable (e.g., flux)
    yerr : array
        Uncertainties in y
    p0 : array
        Initial guess for parameters
    bounds : tuple of arrays, optional
        Lower and upper bounds for parameters
    niter : int
        Number of Monte Carlo iterations (default: 500)
    errfunc : str
        Error estimation method: 'mad' or 'stddev' (default: 'mad')
    return_sample : bool
        Return full MC sample in addition to mean/error (default: False)
    chisq_thresh : float
        Chi-square threshold to filter bad fits (default: np.inf, no filter)
    sig_clip : float
        Sigma clipping threshold for stddev (default: 7.0)
    autocorrelation : bool or int
        - False: assume uncorrelated noise (default)
        - True: estimate correlation length from fit residuals
        - int: use fixed correlation length in pixels
        For spectroscopy, use max_lag=5-10 to focus on line-scale correlations
    max_lag : int
        Maximum lag for ACF estimation (default: 10 pixels)
        Relevant for detecting spurious lines from correlated noise
    baseline_order : int
        Polynomial order for baseline removal in ACF (default: 0)
        Use 0 for residuals, 1-2 for raw spectra with trends
    max_nfev : int
        Max function evaluations per fit (default: 5000)
    
    Returns:
    --------
    If return_sample=False:
        [params, errors] : list of arrays
    If return_sample=True:
        ([params, errors], sample) : tuple
            where sample is (niter, nparams) array of all fitted parameters
    """

    try:
        # Initial fit using large max_nfev
        popt, _ = curve_fit(f, x, y, sigma=yerr, p0=p0, bounds=bounds,
                              absolute_sigma=True, max_nfev=100000)

        # Estimate correlation length from RESIDUALS
        if isinstance(autocorrelation, bool) and autocorrelation:
            residuals = y - f(x, *popt)
            correlation_length, error_inflation = autocorr_length(x, residuals, yerr, max_lag, baseline_order)
            print(f"Estimated correlation length: {correlation_length:.3f} pixels")
            if error_inflation > 1.0:
                print(f"→ Applying error inflation: {error_inflation:.2f}x (for Monte Carlo noise generation)")
                yerr_inflated = yerr * error_inflation
            else:
                yerr_inflated = yerr
                print(f"→ No error inflation needed (median residuals within expected range)")
        elif isinstance(autocorrelation, bool) and not autocorrelation:
            correlation_length = 1
            yerr_inflated = yerr
        elif isinstance(autocorrelation, int):
            correlation_length = autocorrelation
            yerr_inflated = yerr
            print(f"Using fixed correlation length: {correlation_length} pixels")
        else:
            correlation_length = 1
            yerr_inflated = yerr

        poptlist = []
        valid_iters = 0
        total_iters = 0
        max_total_iters = 2 * niter
        
        while valid_iters < niter and total_iters < max_total_iters:
            total_iters += 1
            
            # Generate perturbations (using inflated errors if systematic variance detected)
            yper = (gen_corr_noise(yerr_inflated, correlation_length) if correlation_length > 1
                   else np.random.normal(scale=np.abs(yerr_inflated)))

            try:
                popt_i, _ = curve_fit(f, x, y + yper, sigma=yerr, p0=popt,
                                    bounds=bounds, absolute_sigma=True, max_nfev=max_nfev)

                # Chi-square filtering
                if np.isfinite(chisq_thresh):
                    chisq = np.sum(((f(x, *popt_i) - (y + yper))/yerr)**2) / (len(x) - len(popt_i))
                    if chisq > chisq_thresh:
                        continue

                poptlist.append(popt_i)
                valid_iters += 1
            except RuntimeError:
                print(f"Iteration {valid_iters} failed to converge; skipping.")
                continue
        
        # Warn if we hit the max iteration limit
        if total_iters >= max_total_iters and valid_iters < niter:
            print(f"Warning: Reached maximum iteration limit ({max_total_iters}). "
                  f"Only {valid_iters}/{niter} successful fits obtained.")

        if not poptlist:
            raise RuntimeError("All MC iterations failed to converge")

        return (avgfunc(poptlist, errfunc), poptlist) if return_sample else avgfunc(poptlist, errfunc)

    except Exception as e:
        print(f"Error in fit_mc: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None
    

from typing import TypedDict, Optional, Union

class BootstrapParams(TypedDict, total=False):
    niter: int
    errfunc: str
    chisq_thresh: float
    sig_clip: float
    autocorrelation: Union[bool, int]
    max_lag: int
    baseline_order: Optional[int]
    max_nfev: int


def prep_inputs(initial_guesses, bounds, linename, z_lya):
    """
    Prepares and validates input parameters for fitting routines.
    initial_guesses: dictionary of initial guesses of parameters (LPEAK is required, others optional)
    bounds: dictionary of parameter bounds
    linename: name of the line (needs to be in wavedict)
    z_lya: redshift of Lyman-alpha line (used as sanity check for LPEAK)
    """
    # If LPEAK is not provided, raise error
    if 'LPEAK' not in initial_guesses:
        raise ValueError("Initial guess for LPEAK is required.")
    
    # Get the approximate expected observed wavelength based on z_lya
    if linename not in wavedict:
        raise ValueError(f"Line {linename} not found in wavelength dictionary.")
    expected_wavelength = wavedict[linename] * (1 + z_lya)

    # Calculate the velocity buffer from the bounds if provided, else default to +/- 6.25 Angstroms
    wave_buffer = bounds.get('LPEAK', (initial_guesses['LPEAK'] - 6.25, initial_guesses['LPEAK'] + 6.25))
    wave_buffer = (wave_buffer[1] - wave_buffer[0]) / 2.0

    # If the initial guess for LPEAK is more than 500 km/s away from expected, raise warning
    # and reset to generic offset of -200 km/s
    lpeak_init = initial_guesses['LPEAK']
    delta_v = spectro.wave2vel(lpeak_init, wavedict[linename], z_lya)
    if abs(delta_v) > 500:
        warnings.warn(f"Initial guess for LPEAK ({lpeak_init:.2f} Å) is {delta_v:.1f} km/s \
                       from Lyman alpha ({expected_wavelength:.2f} Å); resetting to -200 km/s offset.")
        initial_guesses['LPEAK'] = spectro.vel2wave(-200, wavedict[linename], z_lya)
        bounds['LPEAK'] = (initial_guesses['LPEAK'] - 6.25, initial_guesses['LPEAK'] + 6.25)

    # Check values of initial guesses and bounds for bad values
    for param, value in initial_guesses.items():
        if not np.isfinite(value):
            raise ValueError(f"Initial guess for {param} is not finite: {value}")
        if param in bounds:
            lower, upper = bounds[param]
            if not (np.isfinite(lower) and np.isfinite(upper)):
                raise ValueError(f"Bounds for {param} are not finite: {bounds[param]}")
            if lower >= upper:
                raise ValueError(f"Lower bound must be less than upper bound for {param}: {bounds[param]}")
            if not (lower <= value <= upper):
                warnings.warn(f"Initial guess for {param} ({value}) is outside bounds {bounds[param]}; adjusting to midpoint.")
                initial_guesses[param] = (lower + upper) / 2.0

    return initial_guesses, bounds


def check_inputs(p0, bounds):
    """Ensure initial guesses are within bounds and not NaN."""
    p0_checked = []
    for i, val in enumerate(p0):
        lower, upper = bounds[0][i], bounds[1][i]
        
        # Check for NaN first
        if np.isnan(val):
            # Use midpoint of bounds for NaN values
            if np.isinf(lower) or np.isinf(upper):
                # Handle infinite bounds
                if np.isinf(lower) and np.isinf(upper):
                    val = 0.0  # Both infinite, use 0
                elif np.isinf(lower):
                    val = upper * 0.5 if upper > 0 else upper - 100
                else:  # upper is inf
                    val = lower * 0.5 if lower < 0 else lower + 100
            else:
                val = (lower + upper) / 2.0
            print(f"WARNING: Initial guess {i} is NaN; adjusting to {val:.3e}")
        elif not (lower <= val <= upper):
            # Out of bounds - use midpoint or adjust slightly inside bounds
            if np.isinf(lower) or np.isinf(upper):
                if np.isinf(lower):
                    val = upper * 0.9 if upper != 0 else upper - 0.1
                else:  # upper is inf
                    val = lower * 1.1 if lower != 0 else lower + 0.1
            else:
                val = (lower + upper) / 2.0
            print(f"WARNING: Initial guess {val} is outside bounds ({lower}, {upper}); adjusting to {val:.3e}")
            
        p0_checked.append(val)
    return p0_checked, bounds
        

def fit_line(wavelength, spectrum, errors, linename, initial_guesses, bounds = {},
             continuum_buffer = 25., plot_result = True, ax_in = None,
             bootstrap_params: Optional[BootstrapParams] = None):
    """
    Fits a single or double gaussian profile to a line (plus its doublet partner if it has one).
    If the line is Lyman alpha, returns the result of a more complex fitting procedure.

    wavelength: wavelength array
    spectrum  : flux density array
    errors    : flux density uncertainties
    linename  : name of the line (needs to be in wavedict)
    initial_guesses: dictionary of initial guesses of parameters (LPEAK is required, others optional)
    bounds    : dictionary of parameter bounds
    continuum_buffer: buffer region around the line to include in the fit
    plot_result: whether to plot the fitting result
    ax_in     : axis to plot on (if plot_result is True)
    bootstrap_params: optional dictionary of parameters for Monte Carlo error estimation
                      (niter, errfunc, chisq_thresh, sig_clip, autocorrelation, max_lag, 
                       baseline_order, max_nfev). If None, uses standard curve_fit errors.
    """

    # If the line is Lyman alpha, use the specialised Lyman alpha fitting function
    if linename == 'LYALPHA':
        raise ValueError("Use specialized Lyman-alpha fitting function.")

    # Get initial guesses
    lpeak_init      = initial_guesses['LPEAK']
    flux_init       = initial_guesses.get('FLUX', 10)
    fluxsecond_init = initial_guesses.get('FLUX2', flux_init) # this will be used if fitting a doublet
    fwhm_init       = initial_guesses.get('FWHM', 3.0)
    cont_init       = initial_guesses.get('CONT', 0)
    slope_init      = initial_guesses.get('SLOPE', 0)

    # Get bounds
    lpeak_bounds = bounds.get('LPEAK', (lpeak_init - 6.25, lpeak_init + 6.25))
    flux_bounds = bounds.get('FLUX', (0, np.inf))
    fluxsecond_bounds = bounds.get('FLUX2', (0, np.inf)) # this will be used if fitting a doublet
    fwhm_bounds = bounds.get('FWHM', (2.4, (300 / c) * lpeak_init))
    cont_bounds = bounds.get('CONT', (-np.inf, np.inf))
    slope_bounds = bounds.get('SLOPE', (-np.inf, np.inf))

    # Determine whether to fit as a single line or doublet
    method = which_fit_method(linename)

    # Rest wavelength of the line
    rest_wavelength = wavedict[linename]
    rest_wavelength_second = np.nan
    if method == 'doublet':
        rest_wavelength_second = wavedict[doubletdict[linename][1]]

    # Define fitting region, masking any problematic areas (zeros, nans, infs)
    fit_mask = generate_spec_mask(wavelength, spectrum, errors, 
                                  lpeak_init, continuum_buffer, linename)
    # Extend it if its a doublet to include the second line
    if method == 'doublet':
        fit_mask += generate_spec_mask(wavelength, spectrum, errors,
                                       lpeak_init * (rest_wavelength_second / rest_wavelength),
                                       continuum_buffer, linename)
    
    # Make sure there are enough good points to fit
    if fit_mask.sum() < 10:
        print(f"Not enough good points to fit {linename}.")
        return {}
        
    # Make sure that the immediate region around the line is not all masked
    line_region = np.logical_and(lpeak_init - 2.5 < wavelength, wavelength < lpeak_init + 2.5)
    # Include second line region if doublet
    if method == 'doublet':
        line_region += np.logical_and(
            lpeak_init * (rest_wavelength_second / rest_wavelength) - 2.5 < wavelength,
            wavelength < lpeak_init * (rest_wavelength_second / rest_wavelength) + 2.5
        )
    if np.sum(fit_mask & line_region) < 3 and not method == 'doublet':
        print(f"Immediate region around {linename} is too heavily masked. Abandoning fit.")
        return {}

    # Confine data to fitting region
    wl_fit   = wavelength[fit_mask]
    spec_fit = spectrum[fit_mask]
    err_fit  = errors[fit_mask]

    # Initialise fit result dictionary that the function will return
    fit_result = {}
    model = None

    # Prepare for fitting
    if method == 'doublet':
        # Define primary and secondary lines of the doublet
        primary   = linename
        secondary = doubletdict[linename][1]

        # Get their wavelength ratio
        primary_rest   = wavedict[primary]
        secondary_rest = wavedict[secondary]
        rest_ratio     = secondary_rest / primary_rest

        model = mdl.gaussian_doublet(rest_ratio)

        # Generate list of initial guesses and bounds
        initg = [flux_init, lpeak_init, fwhm_init, fluxsecond_init, cont_init, slope_init]
        bounds = (
            [flux_bounds[0], lpeak_bounds[0], fwhm_bounds[0],
                fluxsecond_bounds[0], cont_bounds[0], slope_bounds[0]],
                [flux_bounds[1], lpeak_bounds[1], fwhm_bounds[1],
                fluxsecond_bounds[1], cont_bounds[1], slope_bounds[1]]
        )

        # Make sure that initial guesses are within bounds, raise warning if not and default to midpoint
        initg, bounds = check_inputs(initg, bounds)

        # Try fitting multiple times in case of failure
        max_retries = 3
        success = False
        poptg, pcovg = [np.nan for _ in initg], [np.nan for _ in initg] # Default to NaNs

        for attempt in range(max_retries):
            try:
                poptg, pcovg = curve_fit(
                    model, wl_fit, spec_fit,
                    sigma=err_fit, p0=initg, absolute_sigma=True,
                    max_nfev=100000, method='trf', bounds=bounds
                )
                success = True
                break
            except (RuntimeError, ValueError) as e:
                print(f"curve_fit attempt {attempt+1} failed: {e}")
                # Optionally: perturb p0 for next attempt
                # initg = np.array(initg) + np.random.normal(0, 0.1, size=len(initg))
        
        if not success: # If it keeps failing, print warning and return empty dict
            print("All curve_fit attempts failed for doublet fit.")
            return {}
        
        # If bootstrap_params provided, use fit_mc for error estimation
        if bootstrap_params is not None:
            print("Using Monte Carlo error estimation for doublet fit...")
            mc_result = fit_mc(
                model, wl_fit, spec_fit, err_fit, poptg, bounds=bounds,
                niter=bootstrap_params.get('niter', 500),
                errfunc=bootstrap_params.get('errfunc', 'std'),
                chisq_thresh=bootstrap_params.get('chisq_thresh', np.inf),
                sig_clip=bootstrap_params.get('sig_clip', 7.0),
                autocorrelation=bootstrap_params.get('autocorrelation', False),
                max_lag=bootstrap_params.get('max_lag', 5),
                baseline_order=bootstrap_params.get('baseline_order', 0),
                max_nfev=bootstrap_params.get('max_nfev', 5000)
            )
            if mc_result is None:
                print("Monte Carlo error estimation failed, using curve_fit errors")
                error_values = np.sqrt(np.diag(pcovg))
            else:
                poptg = mc_result[0]  # Best-fit parameters from MC
                error_values = mc_result[1]  # Errors from MC
        else:
            error_values = np.sqrt(np.diag(pcovg))
        
        # Populate the parameter dictionary
        param_list = ['FLUX', 'LPEAK', 'FWHM', 'FLUX2', 'CONT', 'SLOPE']
        param_dict = {param: val for param, val
                      in zip(param_list, poptg)}
        error_dict = {param: error_values[i] 
                      for i, param in enumerate(param_list)}
     
        # Calculate the reduced chi-squared value of the fit
        reduced_chisq = np.nansum(np.square((mdl.gaussian_doublet(rest_ratio)(wl_fit, *poptg) -
                                    spec_fit) / err_fit)) / (np.nansum(fit_mask) - len(initg))
        

    elif method == 'single':
        # Generate list of initial guesses and bounds
        initg = [flux_init, lpeak_init, fwhm_init, cont_init, slope_init]
        bounds = (
            [flux_bounds[0], lpeak_bounds[0], fwhm_bounds[0], cont_bounds[0], slope_bounds[0]],
            [flux_bounds[1], lpeak_bounds[1], fwhm_bounds[1], cont_bounds[1], slope_bounds[1]]
        )

        # Make sure that initial guesses are within bounds, raise warning if not and default to midpoint
        initg, bounds = check_inputs(initg, bounds)

        model = mdl.gaussian

        # Try fitting multiple times in case of failure
        max_retries = 3
        success = False
        poptg, pcovg = [np.nan for _ in initg], [np.nan for _ in initg] # Default to NaNs

        for attempt in range(max_retries):
            try:
                poptg, pcovg = curve_fit(
                    model, wl_fit, spec_fit,
                    sigma=err_fit, p0=initg, absolute_sigma=True,
                    max_nfev=100000, method='trf', bounds=bounds
                )
                success = True
                break
            except (RuntimeError, ValueError) as e:
                print(f"curve_fit attempt {attempt+1} failed: {e}")
                # Optionally: perturb p0 for next attempt
                # initg = np.array(initg) + np.random.normal(0, 0.1, size=len(initg))
        
        if not success: # If it keeps failing, print warning and return empty dict
            print("All curve_fit attempts failed for single line fit.")
            return {}
        
        # If bootstrap_params provided, use fit_mc for error estimation
        if bootstrap_params is not None:
            print("Using Monte Carlo error estimation for single line fit...")
            mc_result = fit_mc(
                model, wl_fit, spec_fit, err_fit, poptg, bounds=bounds,
                niter=bootstrap_params.get('niter', 500),
                errfunc=bootstrap_params.get('errfunc', 'std'),
                chisq_thresh=bootstrap_params.get('chisq_thresh', np.inf),
                sig_clip=bootstrap_params.get('sig_clip', 7.0),
                autocorrelation=bootstrap_params.get('autocorrelation', False),
                max_lag=bootstrap_params.get('max_lag', 5),
                baseline_order=bootstrap_params.get('baseline_order', 0),
                max_nfev=bootstrap_params.get('max_nfev', 5000)
            )
            if mc_result is None:
                print("Monte Carlo error estimation failed, using curve_fit errors")
                error_values = np.sqrt(np.diag(pcovg))
            else:
                poptg = mc_result[0]  # Best-fit parameters from MC
                error_values = mc_result[1]  # Errors from MC
        else:
            error_values = np.sqrt(np.diag(pcovg))
        
        # Populate the parameter dictionary
        param_list = ['FLUX', 'LPEAK', 'FWHM', 'CONT', 'SLOPE']
        param_dict = {param: val for param, val
                      in zip(param_list, poptg)}
        error_dict = {param: error_values[i] 
                      for i, param in enumerate(param_list)}
     
        # Calculate the reduced chi-squared value of the fit
        reduced_chisq = np.nansum(np.square((mdl.gaussian(wl_fit, *poptg) -
                                    spec_fit) / err_fit)) / (np.nansum(fit_mask) - len(initg))

    else:
        raise ValueError(f"Unknown fitting method: {method}")
        
    # If the reduced chi-squared is very high, print warning
    if reduced_chisq > 3.0:
        print(f'WARNING: HIGH REDUCED CHI SQUARED STATISTIC ({reduced_chisq})! REVIEW RESULT.')
    
    # Check for multiple comparable peaks in residuals
    residuals = spec_fit - model(wl_fit, *poptg)
    fitted_amplitude = poptg[0]  # First parameter is always flux/amplitude
    fitted_center = poptg[1]     # Second parameter is always center
    fitted_fwhm = poptg[2]       # Third parameter is always FWHM
    n_params = len(poptg)
    
    peak_check = check_multiple_peaks(
        wl_fit, residuals, err_fit,
        fitted_center, fitted_amplitude, fitted_fwhm,
        n_fit_params=n_params,
        chi2_threshold=3.0
    )
    
    if peak_check['suspicious']:
        print(f"WARNING: {peak_check['message']}")
        for i, peak in enumerate(peak_check['peak_info']):
            print(f"  Suspicious peak {i+1}: λ={peak['wavelength']:.2f}, "
                  f"SNR={peak['snr']:.1f}, amp_ratio={peak['amplitude_ratio']:.2f}")

    # Populate the fit_result dictionary
    fit_result['method'] = method
    fit_result['popt'] = poptg
    fit_result['pcov'] = pcovg
    fit_result['reduced_chisq'] = reduced_chisq
    fit_result['fit_mask'] = fit_mask
    fit_result['wl_fit'] = wl_fit
    fit_result['spec_fit'] = spec_fit
    fit_result['err_fit'] = err_fit
    fit_result['param_dict'] = param_dict
    fit_result['error_dict'] = error_dict
    fit_result['model'] = model
    fit_result['peak_check'] = peak_check  # Add multiple peaks check result
    fit_result['multipeak_flag'] = peak_check['flag']  # Quick access to flag

    # If requested, plot the fitting result
    if plot_result:

        if ax_in is None:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='w')
        else:
            ax = ax_in

        # Plot the data
        ax.plot(wl_fit, spec_fit, drawstyle='steps-mid', color='black', alpha=0.7, label='Fitted Data')
        ax.fill_between(wl_fit, spec_fit - err_fit, spec_fit + err_fit, step='mid', color='grey', alpha=0.5)

        # Generate a finely sampled wavelength array for plotting the model
        wl_model = np.linspace(np.min(wl_fit), np.max(wl_fit), 1000)

        # Overplot the fitted model
        if method == 'doublet' and success:
            model_flux = mdl.gaussian_doublet(rest_ratio)(wl_model, *poptg)
            ax.plot(wl_model, model_flux, color='red', label='Doublet Fit', lw=2)
            # Plot individual components
            primary_comp = mdl.gaussian(wl_model, poptg[0], poptg[1], poptg[2], 0, 0)
            secondary_comp = mdl.gaussian(wl_model, poptg[3], poptg[1]*rest_ratio, poptg[2], 0, 0)
            ax.plot(wl_model, primary_comp, color='orange', ls='--', label='Primary Component')
            ax.plot(wl_model, secondary_comp, color='green', ls='--', label='Secondary Component')
        elif method == 'single' and success:
            model_flux = mdl.gaussian(wl_model, *poptg)
            ax.plot(wl_model, model_flux, color='red', label='Single Line Fit', lw=2)

        ax.set_xlabel(r'Wavelength (\AA)')
        ax.set_ylabel('Flux Density')
        ax.set_title(f'Fit to {linename} Line')
        ax.legend()
        plt.show() if ax_in is None else None
        plt.close() if ax_in is None else None

    return fit_result


def refit_other_line(wave, spec, spec_err, row, line_tab_row = None, width=25, ax_in=None,
                     line_name = None, bootstrap_params: Optional[BootstrapParams] = None):
    """Refit a non-Lyman alpha emission line based on prior fitting results.
    Uses a single Gaussian plus linear baseline model with initial guesses from the table row.
    In cases where no significant fit was previously found, gets initial guesses from the R21 catalogue.
    If a secondary line of a doublet is passed, the primary is inferred and passed to the fitting function instead.
    
    Parameters:
    -----------
    wave:         wavelength array
    spec:         flux density array
    spec_err:     flux density error array
    row:          the row of the megatab containing the fitting results to use as priors
    line_tab_row: the row of the line table containing information about the line to fit from the 
                  original catalogues. Only needs to be supplied if no significant fit was found previously.
    width:        the width (in Angstroms) around the line to use for fitting
    bootstrap_params: optional dictionary of parameters for Monte Carlo error estimation
                      (niter, errfunc, chisq_thresh, sig_clip, autocorrelation, max_lag, 
                       baseline_order, max_nfev). If None, uses standard curve_fit errors.
    
    Returns:
    --------
    param_dict : dict
        Dictionary of fitted parameters (FLUX, LPEAK, FWHM, CONT, SLOPE, and FLUX2 if doublet)
    error_dict : dict
        Dictionary of parameter uncertainties
    model : callable
        The model function used for fitting
    reduced_chisq : float
        Reduced chi-square of the fit
    multipeak_flag : str
        Quality flag: 'm' if multiple comparable peaks detected, '' otherwise
    """
    if line_tab_row is None and line_name is None:
        raise ValueError("Either line_tab_row or line_name must be provided.")
    if line_tab_row is not None:
        line_name = line_tab_row['LINE']

    # Was a significant fit found previously?
    significant_fit = np.abs(row['SNR_'+line_name]) > 3.0

    # Is the line in a doublet?
    doublet = np.any(const.flines == line_name) or np.any(const.slines == line_name)
    # Get the doublet ratio if so, otherwise set to 1.0
    doublet_ratio = 1.0
    primary_line = line_name
    secondary_line = None
    if doublet:
        if np.any(const.flines == line_name):
            secondary_line = const.doublets[line_name][1]
            doublet_ratio = const.wavedict[secondary_line] / const.wavedict[primary_line]
        elif np.any(const.slines == line_name):
            # Change the primary line name to the first line in the doublet
            primary_line = const.slines[np.where(const.slines == line_name)[0][0] - 1]
            secondary_line = line_name
            doublet_ratio = const.wavedict[secondary_line] / const.wavedict[primary_line]
    # What kind of function do we need to use, single or double Gaussian?
    model_func = mdl.gaussian_doublet(doublet_ratio) if doublet else mdl.gaussian

    # Calculate observed wavelength, multiplying by the ratio of the primary line rest wavelength to this line's rest wavelength
    # (this value will be 1.0 if this is the primary line)
    if line_tab_row is None:
        r21_observed_wavelength = row[f'LPEAK_{primary_line}']
        r21_observed_flux      = row[f'FLUX_{primary_line}']
    else:
        r21_observed_wavelength = line_tab_row['LBDA_OBS'] * const.wavedict[primary_line] / const.wavedict[line_name]
        r21_observed_flux      = line_tab_row['FLUX']

    print(f"Refitting {primary_line} at {r21_observed_wavelength:.2f} Angstroms for {row['CLUSTER']} {row['iden']}...")
    
    # Names of parameters to be fitted
    param_names = ['FLUX', 'LPEAK', 'FWHM']
    
    # Initial guesses from the table, or the R21 catalogue if not available
    flux_init = row[f'FLUX_{primary_line}'] if significant_fit else r21_observed_flux
    cen_init  = row[f'LPEAK_{primary_line}'] if significant_fit else r21_observed_wavelength
    wid_init  = row[f'FWHM_{primary_line}'] if significant_fit else 3.0 # just use a generic value here
    # Put these into p0
    p0 = [flux_init, cen_init, wid_init]
    # And set the corresponding bounds
    bounds = (
        [-10000, cen_init - 6.25, 2.4],  # lower bounds, with width not less than the spectral resolution
        [10000 , cen_init + 6.25, (300. / c) * cen_init ]   # upper bounds, with width not exceeding 300km/s
    )

    # If the line is a doublet, we need to add the flux of the second component (other parameters are tied)
    if doublet:
        # Check if the secondary line was also fitted (not NaN)
        secondary_fit = significant_fit and not np.isnan(row[f'FLUX_{secondary_line}'])
        flux_init_2 = row[f'FLUX_{secondary_line}'] if secondary_fit else flux_init
        param_names.extend(['FLUX2'])
        p0.extend([flux_init_2])
        # Also extend the bounds
        bounds[0].extend([-10000])
        bounds[1].extend([10000])

    # Add continuum and slope initial guesses
    cont_init = row[f'CONT_{primary_line}'] if significant_fit else np.nanmedian(spec) # use median of spectrum
    slope_init = row[f'SLOPE_{primary_line}'] if significant_fit else 0.0  # assume flat continuum initially
    p0.extend([cont_init, slope_init])
    bounds[0].extend([-50, -1000])
    bounds[1].extend([2000, 1000])
    param_names.extend(['CONT', 'SLOPE'])

    # Create initial guesses and bounds dictionaries
    initial_guesses = dict(zip(param_names, p0))
    bounds_dict = {param : (bounds[0][i], bounds[1][i]) for i, param in enumerate(param_names)}
    
    # fit_line will handle validation of initial guesses and bounds
    fit = fit_line(wave, spec, spec_err, primary_line, initial_guesses, bounds=bounds_dict,
                   continuum_buffer=width, plot_result=True, ax_in=ax_in,
                   bootstrap_params=bootstrap_params)
    
    if 'param_dict' not in fit:
        print(f"Fit failed for {primary_line} in {row['CLUSTER']} {row['iden']}.")
        nandict = {param: np.nan for param in initial_guesses.keys()}
        return nandict, nandict, None, np.nan, ''
    else:
        multipeak_flag = fit.get('multipeak_flag', '')
        return fit['param_dict'], fit['error_dict'], fit['model'], fit['reduced_chisq'], multipeak_flag


def flatten_spectrum(spectrum, return_continuum=False):
    """
    Remove a linear continuum trend from a spectrum.
    
    This function fits a straight line to the spectrum and subtracts the slope,
    leaving only the continuum level. This is useful for absorption line analysis
    where you want to remove large-scale continuum variations.
    
    Parameters
    ----------
    spectrum : numpy.ndarray
        1D array of flux values to flatten.
    return_continuum : bool, optional
        If True, return both the flattened spectrum and the fitted continuum model.
        If False, return only the flattened spectrum. Default is False.
    
    Returns
    -------
    flattened : numpy.ndarray
        Spectrum with linear trend removed, retaining only the continuum level.
    continuum : numpy.ndarray (only if return_continuum=True)
        The fitted linear continuum model.
    
    Notes
    -----
    - Uses scipy.optimize.curve_fit with a simple linear model: f(x) = a*x + b
    - The returned spectrum has the slope removed but keeps the continuum level (b)
    - This preserves the approximate flux level while removing linear trends
    - Useful for normalizing absorption features before stacking
    
    Examples
    --------
    >>> from astro_utils import fitting as aufit
    >>> import numpy as np
    >>> # Create a spectrum with linear trend
    >>> x = np.arange(100)
    >>> spec = 10.0 + 0.05 * x + np.random.normal(0, 0.1, 100)
    >>> # Flatten it
    >>> flat_spec = aufit.flatten_spectrum(spec)
    >>> # Or get the continuum model too
    >>> flat_spec, continuum = aufit.flatten_spectrum(spec, return_continuum=True)
    """
    
    def linear(x, a, b):
        """Simple linear function for continuum fitting."""
        return a * x + b
    
    # Create pixel array
    pixels = np.arange(len(spectrum))
    
    # Fit linear function to the spectrum
    popt, pcov = curve_fit(linear, pixels, spectrum, p0=[0, 0])
    
    # Calculate the continuum model
    continuum_model = linear(pixels, *popt)
    
    # Remove the slope but keep the continuum level
    # This is: spectrum - (a*x + b) + b = spectrum - a*x
    flattened = spectrum - continuum_model + popt[1]
    
    if return_continuum:
        return flattened, continuum_model
    else:
        return flattened
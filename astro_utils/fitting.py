
from . import constants as const
import numpy as np
from scipy.optimize import curve_fit
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

def autocorr_length(wave, spec, max_lag=10, baseline_order=None):
    """Estimate correlation length by fitting exponential decay to ACF.
    
    Returns the e-folding length τ where ACF(τ) = 1/e ≈ 0.368.
    This is more principled than threshold crossing.
    """
    # 1. Baseline removal with diagnostics
    if baseline_order is not None:
        p = nppoly.fit(wave, spec, baseline_order)
        residuals = spec - p(wave)
    else:
        residuals = spec - np.mean(spec)

    residuals = residuals - np.mean(residuals)

    # 2. ACF calculation with normalization check
    acf = np.correlate(residuals, residuals, mode='full')[len(residuals)-1:]
    if acf[0] <= 0:
        return 1  # Protection against bad normalization
    acf = acf / acf[0]

    # 3. Estimate noise floor from tail
    tail_start = min(max_lag + 1, len(acf)-20)
    tail_acf = acf[tail_start:tail_start+20]
    noise_floor_mean = np.mean(tail_acf)
    noise_floor_std = np.std(tail_acf)
    
    # Conservative threshold: 90th percentile of noise distribution (1.28 sigma)
    threshold = max(0.1, noise_floor_mean + 1.28 * noise_floor_std)
    
    print(f"\nACF[:10]: {acf[1:11]}")
    print(f"Noise floor: {noise_floor_mean:.3f} +/- {noise_floor_std:.3f}")
    print(f"Threshold (90th percentile): {threshold:.3f}")
    
    # 4. Find where ACF drops below threshold (fallback method)
    crossing_lag = None
    for k in range(1, min(max_lag, len(acf)-1)):
        if acf[k] < threshold:
            crossing_lag = k
            break
    
    # 5. Fit exponential decay ACF(k) = exp(-k/τ) to estimate τ
    # Only fit to points above threshold
    if crossing_lag is not None:
        fit_range = crossing_lag + 2  # Fit a bit beyond crossing
    else:
        fit_range = max_lag
    
    lags = np.arange(1, min(fit_range, len(acf)))
    acf_to_fit = acf[1:fit_range]
    
    # Only fit positive ACF values (take log)
    valid_mask = acf_to_fit > threshold
    if np.sum(valid_mask) >= 2:  # Need at least 2 points to fit
        try:
            # Fit log(ACF) = -k/τ  =>  slope = -1/τ
            # Weight by 1/sqrt(k) to give more weight to early lags
            weights = 1.0 / np.sqrt(lags[valid_mask])
            coeffs = np.polyfit(lags[valid_mask], np.log(acf_to_fit[valid_mask]), 
                               deg=1, w=weights)
            tau_fit = -1.0 / coeffs[0]
            
            # Sanity check: tau should be positive and reasonable
            if 0.5 <= tau_fit <= 2 * max_lag:
                print(f"Fitted τ = {tau_fit:.2f} pixels (exponential decay)")
                if crossing_lag is not None:
                    print(f"Threshold crossing at lag {crossing_lag} (ACF={acf[crossing_lag]:.3f})")
                return max(1.0, tau_fit)  # Return at least 1
            else:
                print(f"Fitted τ = {tau_fit:.2f} outside reasonable range, using fallback")
        except (np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Exponential fit failed: {e}, using fallback")
    
    # 6. Fallback to threshold crossing or max_lag
    if crossing_lag is not None:
        print(f"Using threshold crossing at lag {crossing_lag} (ACF={acf[crossing_lag]:.3f})")
        return crossing_lag
    else:
        print(f"WARNING: No crossing found within max_lag={max_lag} - returning max_lag")
        if acf[max_lag] > 2 * noise_floor_std:
            print(f"WARNING: ACF({max_lag}) = {acf[max_lag]:.3f} still above noise floor")
        return max_lag

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
    """Enhanced MC fitting with robust correlation handling.
    f              : model function to fit
    x              : independent variable data
    y              : dependent variable data
    yerr           : uncertainties in y
    p0             : initial guess for the parameters
    bounds         : bounds for the parameters (default: None)
    niter          : number of Monte Carlo iterations (default: 1000)
    errfunc        : function to estimate parameter uncertainties ('stddev' or 'mad', default: 'stddev')
    return_sample  : whether to return the full sample of fitted parameters (default: False)
    chisq_thresh   : chi-square threshold to filter fits (default: np.inf, no filtering)
    sig_clip       : sigma clipping threshold for stddev calculation (default: 7.0)
    autocorrelation: whether to estimate correlation length from residuals (default: False)
                     or provide a fixed integer value for correlation length
    max_lag        : maximum lag to consider in autocorrelation (default: 10)
    baseline_order : order of polynomial baseline to subtract before autocorrelation (default: 0)
    max_nfev       : maximum number of function evaluations for curve_fit (default: 10000)
    """

    try:
        # Initial fit using large max_nfev
        popt, _ = curve_fit(f, x, y, sigma=yerr, p0=p0, bounds=bounds,
                              absolute_sigma=True, max_nfev=100000)

        # Estimate correlation length from RESIDUALS
        if isinstance(autocorrelation, bool) and autocorrelation:
            residuals = y - f(x, *popt)
            correlation_length = autocorr_length(x, residuals, max_lag, baseline_order)
            print(f"Estimated correlation length: {correlation_length} pixels")
        elif isinstance(autocorrelation, bool) and not autocorrelation:
            correlation_length = 1
        elif isinstance(autocorrelation, int):
            correlation_length = autocorrelation
            print(f"Using fixed correlation length: {correlation_length} pixels")
        else:
            correlation_length = 1

        poptlist = []
        valid_iters = 0
        while valid_iters < niter:
            # Generate perturbations
            yper = (gen_corr_noise(yerr, correlation_length) if correlation_length > 1
                   else np.random.normal(scale=np.abs(yerr)))

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
    Returns a dictionary of optimal parameters and their associated uncertainties.
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
        return nandict, nandict, None, np.nan
    else:
        return fit['param_dict'], fit['error_dict'], fit['model'], fit['reduced_chisq']

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

# Import dictionary of line wavelengths and speed of light
wavedict    = const.wavedict
doubletdict = const.doublets
skylinedict = const.skylines
c           = const.c  # speed of light in km/s

def mask_skylines(wavelength):
    """
    Masks out regions around known sky lines to avoid contamination in fits.

    wavelength: wavelength array
    lambda_obs: observed wavelength of the line
    continuum_buffer: buffer region around the line to include in the fit
    """
    sky_mask = np.ones_like(wavelength, dtype=bool)

    for skyline in skylinedict.values():
        sky_mask &= ~((wavelength > (skyline - 2.5)) & (wavelength < (skyline + 2.5)))

    return sky_mask

def mask_otherlines(wavelength, expected_wavelength, linename):
    """
    Mask out regions around other known lines to avoid contamination in fits.
    
    wavelength: wavelength array
    linename  : name of the line (needs to be in wavedict)
    """
    # Get all other lines except the one being fitted and its doublet partner (if any)
    otherlines = [line for line in wavedict.keys() 
                  if (line != linename and line not in doubletdict.get(linename, ()) 
                      and linename not in doubletdict.get(line, ()))]
    
    # Initialize mask to all True
    line_mask = np.ones_like(wavelength, dtype=bool)

    # Get the expected redshift
    z_exp = (expected_wavelength / wavedict[linename]) - 1

    # Mask out +/- 2.5 Angstroms around each other line
    for line in otherlines:
        line_center = wavedict[line] * (1 + z_exp)
        line_mask &= ~((wavelength > (line_center - 2.5)) & (wavelength < (line_center + 2.5)))

    return line_mask

def generate_spec_mask(wavelength, spectrum, errors, lpeak_init, continuum_buffer, linename):
    """
    Generates a mask for the fitting region around a specified observed wavelength,
    excluding problematic areas (zeros, nans, infs), skylines, and unrelated spectral lines.

    wavelength: wavelength array
    spectrum  : flux density array
    errors    : flux density uncertainties
    lpeak_init: initial guess for the observed wavelength of the line
    continuum_buffer: buffer region around the line to include in the fit
    linename   : name of the line (needs to be in wavedict)
    """
    # Fitting region +/- continuum_buffer around the line
    fit_mask = (wavelength > (lpeak_init - continuum_buffer)) & (wavelength < (lpeak_init + continuum_buffer))

    # Mask out problematic data points
    fit_mask &= ~np.isnan(wavelength)
    fit_mask &= ~np.isnan(spectrum)
    fit_mask &= ~np.isnan(errors)
    fit_mask &= spectrum != 0
    fit_mask &= (errors > 0)
    fit_mask &= ~np.isinf(spectrum)
    fit_mask &= ~np.isinf(errors)

    # Mask out skylines and other known lines
    fit_mask &= mask_skylines(wavelength)
    fit_mask &= mask_otherlines(wavelength, lpeak_init, linename)

    return fit_mask

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
    """More reliable correlation length estimator."""
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

    # 3. Dynamic thresholding with diagnostics
    tail_start = min(max_lag + 1, len(acf)-20)
    tail_acf = acf[tail_start:tail_start+20]
    threshold = 0.33  # More sensitive threshold

    print(f"\nACF[:10]: {acf[1:11]}")
    print(f"Noise floor std: {np.std(tail_acf):.3f}")
    print(f"Using threshold: {threshold:.3f}")

    # 4. More lenient crossing condition
    for k in range(1, min(max_lag, len(acf)-1)):
        if acf[k] < threshold:
            print(f"Found crossing at lag {k} (ACF={acf[k]:.3f})")
            return k

    return max_lag

def gen_corr_noise(yerr, corr_len, size=None):
    """Simpler, more robust correlated noise generator.
    Args:
        yerr: Array of error values
        corr_len: Desired correlation length in pixels (>1)
        size: Optional output size (defaults to len(yerr))
    Returns:
        Correlated noise array with same shape as yerr/size
    """
    if corr_len <= 1:
        return np.random.normal(scale=yerr, size=size)

    n = len(yerr) if size is None else size

    # Simplified AR(1) process with guaranteed correlations
    noise = np.zeros(n)
    noise[0] = np.random.normal()

    # Stronger persistence factor
    phi = np.exp(-1.0/(corr_len-1))

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
           autocorrelation=False, max_lag=5, baseline_order=0, max_nfev=5000):
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
    max_lag        : maximum lag to consider in autocorrelation (default: 5)
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
    """Ensure initial guesses are within bounds."""
    p0_checked = []
    for i, val in enumerate(p0):
        lower, upper = bounds[0][i], bounds[1][i]
        if not (lower <= val <= upper):
            print(f"Initial guess {val} is outside bounds ({lower}, {upper}); adjusting to midpoint.")
            val = (lower + upper) / 2.0
        p0_checked.append(val)
    return p0_checked, bounds
        

def fit_line(wavelength, spectrum, errors, linename, initial_guesses, bounds = {},
             continuum_buffer = 25., plot_result = True, ax_in = None):
    """
    Fits a single or double gaussian profile to a line (plus its doublet partner if it has one).
    If the line is Lyman alpha, returns the result of a more complex fitting procedure.

    wavelength: wavelength array
    spectrum  : flux density array
    errors    : flux density uncertainties
    linename  : name of the line (needs to be in wavedict)
    initial_guesses: dictionary of initial guesses of parameters (LPEAK is required, others optional)
    continuum_buffer: buffer region around the line to include in the fit
    plot_result: whether to plot the fitting result
    ax_in     : axis to plot on (if plot_result is True)
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
        
        # If the fit went OK, populate the parameter dictionary
        param_list = ['FLUX', 'LPEAK', 'FWHM', 'FLUX2', 'CONT', 'SLOPE']
        param_dict = {param: val for param, val
                      in zip(param_list, poptg)}
        error_dict = {param: np.sqrt(np.diag(pcovg))[i] 
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
        
        # If the fit went OK, populate the parameter dictionary
        param_list = ['FLUX', 'LPEAK', 'FWHM', 'CONT', 'SLOPE']
        param_dict = {param: val for param, val
                      in zip(param_list, poptg)}
        error_dict = {param: np.sqrt(np.diag(pcovg))[i] 
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
        



class LyaProfile:
    """Class to fit Lyman-alpha emission line profiles using asymmetric Gaussian functions.
    Supports both single-peak and double-peak profiles, as well as constant, linear or 
    damped Lyman alpha baselines.
    """

    # Define parameter order and bounds for single and double peak models (class-level defaults)
    lya_param_order_default = {
        1: ['AMPR', 'LPEAKR', 'DISPR', 'ASYMR', 'CONT'],
        2: ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB', 'AMPR', 'LPEAKR', 'DISPR', 'ASYMR', 'CONT']
    }
    lya_bounds_default = {
        1: ([0,      1215, 0.1,   -0.1, -np.inf],
            [np.inf, 1225, 10,     0.1,  np.inf]),
        2: ([0,      4600, 1.25,  -0.5,  0,      4600, 1.25,  -0.5, -50],
            [1e7,    9350, np.inf, 0.5,  1e7,    9350, np.inf, 0.5,  1e4])
    } # These are hard coded and suitable for MUSE spectra -- modify as needed


    def __init__(self, paramdict, uncertainties):
        """Initialize the lya_profile class using dictionaries of parameters and their uncertainties.
        paramdict: Dictionary of initial parameter guesses.
        uncertainties: Dictionary of uncertainties for the parameters.
        """
        # Determine if single or double peak based on presence of LPEAKB
        self.ncomp = 2 if 'LPEAKB' in paramdict.keys() else 1
        # Determine baseline type based on presence of SLOPE or TAU
        self.baseline = 'lin' if 'SLOPE' in paramdict.keys() else 'const'
        self.baseline = 'damp' if 'TAU' in paramdict.keys() else self.baseline

        # Make per-instance copies of param order and bounds to avoid shared mutation
        self.lya_param_order = {
            1: list(self.__class__.lya_param_order_default[1]),
            2: list(self.__class__.lya_param_order_default[2])
        }
        self.lya_bounds = {
            1: [list(self.__class__.lya_bounds_default[1][0]), list(self.__class__.lya_bounds_default[1][1])],
            2: [list(self.__class__.lya_bounds_default[2][0]), list(self.__class__.lya_bounds_default[2][1])]
        }

        # Extend the lya_param_order and lya_bounds if needed
        if self.baseline == 'lin':
            self.lya_param_order[self.ncomp].extend(['SLOPE'])
            self.lya_bounds[self.ncomp][0].append(-np.inf)
            self.lya_bounds[self.ncomp][1].append(np.inf)
        elif self.baseline == 'damp':
            self.lya_param_order[self.ncomp].extend(['TAU', 'FWHM', 'LPEAK_ABS'])
            self.lya_bounds[self.ncomp][0].extend([0   , -10, 1215.67 - 2.5])
            self.lya_bounds[self.ncomp][1].extend([1000,  20, 1215.67 + 2.5])

        # Populate parameter and uncertainty dictionaries with only relevant parameters
        self.param_dict = {k: paramdict[k] for k in self.lya_param_order[self.ncomp]}
        self.err_dict   = {k: uncertainties[k] for k in self.lya_param_order[self.ncomp]}
        # Select appropriate model function
        if self.baseline == 'lin':
            self.func = mdl.lya_speak_lin if self.ncomp == 1 else mdl.lya_dpeak_lin
        elif self.baseline == 'damp':
            self.func = mdl.lya_speak_damp if self.ncomp == 1 else mdl.lya_dpeak_damp
        elif self.baseline == 'const':
            self.func = mdl.lya_speak if self.ncomp == 1 else mdl.lya_dpeak
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline}")

        # Compute advanced parameters and errors immediately
        # Use Complex for error propagation if uncertainties are provided
        try:
            popt = [self.param_dict[k] for k in self.param_dict.keys()]
            perr = [self.err_dict[k] for k in self.err_dict.keys()]
            complex_pars = [Complex(val, err) for val, err in zip(popt, perr)]
            complex_dict = {k: v for k, v in zip(self.param_dict.keys(), complex_pars)}
            advpars = self.get_adv_params(complex_dict)
            self.adv_params = {k: v.value for k, v in advpars.items()}
            self.adv_errors = {k: v.error for k, v in advpars.items()}
        except Exception:
            self.adv_params = {}
            self.adv_errors = {}
    
    def get_adv_params(self, param_dict):
        """Calculate advanced parameters such as FWHM and integrated fluxes from the fit parameters.
        Returns a dictionary of advanced parameters.
        """
        advdict = {} # Dictionary to hold advanced parameters

        # Calculate FWHM
        if self.ncomp == 2:
            advdict['FWHMB'] = 2 * np.sqrt(2 * np.log(2)) * param_dict['DISPB'] / (1. - 2 * np.log(2) * param_dict['ASYMB'] ** 2.)
        advdict['FWHMR'] = 2 * np.sqrt(2 * np.log(2)) * param_dict['DISPR'] / (1. - 2 * np.log(2) * param_dict['ASYMR'] ** 2.)
        
        # Calculate integrated fluxes over a minimum intrinsic range (4 Angstroms rest-frame)
        intrange_rest = 4.0
        intrange = intrange_rest * (1. + self.param_dict['LPEAKR'] / 1215.67) # Observed frame
        if self.ncomp == 2: # Integrate blue component if present
            asymfrac = np.abs(mdl.lya_swhm(param_dict['DISPB'], param_dict['ASYMB'], -1) \
                              / mdl.lya_swhm(param_dict['DISPB'], param_dict['ASYMB'], +1))
            intstart = param_dict['LPEAKB'] - asymfrac * intrange / (1. + asymfrac)
            intend = param_dict['LPEAKB'] + intrange / (1. + asymfrac)
            xaxis = np.arange(intstart, intend, 0.125)
            modfuncb = mdl.lya_speak(
                xaxis, 
                param_dict['AMPB'], 
                param_dict['LPEAKB'], 
                param_dict['DISPB'], 
                param_dict['ASYMB'], 
                0.
            )
            posmask = modfuncb > 0.
            advdict['FLUXB'] = np.trapz(modfuncb[posmask], x = xaxis[posmask])
        asymfrac = np.abs(mdl.lya_swhm(param_dict['DISPR'], param_dict['ASYMR'], -1) \
                          / mdl.lya_swhm(param_dict['DISPR'], param_dict['ASYMR'], +1))
        intstart = param_dict['LPEAKR'] - asymfrac * intrange / (1. + asymfrac)
        intend = param_dict['LPEAKR'] + intrange / (1. + asymfrac)
        xaxis = np.arange(intstart, intend, 0.125)
        modfuncr = mdl.lya_speak(
            xaxis, 
            param_dict['AMPR'], 
            param_dict['LPEAKR'], 
            param_dict['DISPR'], 
            param_dict['ASYMR'], 
            0.
        )
        posmask = modfuncr > 0.
        advdict['FLUXR'] = np.trapz(modfuncr[posmask], x = xaxis[posmask])
        return advdict
    
    def fit_to(self, x, y, yerr, mask = None, bounds = None, use_bootstrap = False,
               bootstrap_params: Optional[BootstrapParams] = None):
        """Fit the Lyman-alpha profile to the provided data. Returns dictionaries of fit results and uncertainties."""
        if mask is None:
            mask = np.ones(np.size(y))
        if bounds is None:
            bounds = self.lya_bounds[self.ncomp]

        popt, perrs = None, None
        advpars = None
        reduced_chisq = None

        # Use fit_mc for bootstrapping, passing only user-provided params
        if use_bootstrap:
            fit_mc_kwargs = {
                'f': self.func,
                'x': x[mask],
                'y': y[mask],
                'yerr': yerr[mask],
                'p0': list(self.param_dict.values()),
                'bounds': bounds,
                'return_sample': True
            }
            if bootstrap_params:
                fit_mc_kwargs.update(bootstrap_params)

            fitres = fit_mc(**fit_mc_kwargs)

            if fitres is None:
                print("Fitting failed.")
                return None, None, None, None
            fitted_params, distributions = fitres
            popt = np.array(fitted_params[0])
            perrs = np.array(fitted_params[1])
            
            # Make a dictionary to pass to get_adv_params (does not need to be Complex here)
            param_dicts = []
            for pset in distributions:
                param_dicts.append({k: v for k, v in zip(self.param_dict.keys(), pset)})
            
            # Calculate advanced parameters for each distribution and then get median and errors
            advparams_distr = [self.get_adv_params(p) for p in param_dicts]
            adv_params = {}
            adv_errors = {}
            for key in advparams_distr[0].keys():
                vals = np.array([d[key] for d in advparams_distr])
                med, err = avgfunc(vals, bootstrap_params.get('errfunc', 'mad') if bootstrap_params else 'mad')
                adv_params[key] = med
                adv_errors[key] = err

            # Calculate reduced chi-squared of the best fit
            residuals = y[mask] - self.func(x[mask], *popt)
            reduced_chisq = np.nansum((residuals / yerr[mask])**2) / (np.sum(mask) - len(popt))
        else:
            max_retries = 3
            success = False
            popt, pcov = None, None
            for attempt in range(max_retries):
                try:
                    popt, pcov = curve_fit(
                        self.func, x[mask], y[mask], sigma=yerr[mask], absolute_sigma=True,
                        p0=list(self.param_dict.values()), method='trf', max_nfev=100000, bounds=bounds
                    )
                    success = True
                    break
                except (RuntimeError, ValueError) as e:
                    print(f"curve_fit attempt {attempt+1} failed: {e}")
            if not success:
                print("All curve_fit attempts failed.")
                return None, None, None, None
            perrs = np.sqrt(np.diag(pcov))
            complex_pars = [Complex(val, err) for val, err in zip(popt, perrs)]
            advpars = self.get_adv_params(complex_pars)
            adv_params = {k: v.value for k, v in advpars.items()}
            adv_errors = {k: v.error for k, v in advpars.items()}
            residuals = y[mask] - self.func(x[mask], *popt)
            reduced_chisq = np.nansum((residuals / yerr[mask])**2) / (np.sum(mask) - len(popt))

        # Prepare output dictionaries
        fit_params = {k: v for k, v in zip(self.param_dict.keys(), popt)} if popt is not None else {}
        fit_errors = {k: v for k, v in zip(self.param_dict.keys(), perrs)} if perrs is not None else {}
        # Merge advanced params
        fit_params.update(adv_params)
        fit_errors.update(adv_errors)

        # Update instance attributes to reflect latest fit
        self.adv_params = adv_params
        self.adv_errors = adv_errors

        return fit_params, fit_errors, self.func, reduced_chisq
    

def fit_lya_complete(wave, spec, spec_err, row, width=50, plot_result = True,
                   mc_niter=500, rchsq_thresh=2.0, save_plots = False, plot_dir = './'):
    """Fit the Lyman alpha line based on prior fitting results (if present). Here,
    we compare single and double-peaked fits and return the best one based on reduced chi-squared.
    Baselines are handled in increasing order of complexity (const, lin, damp).
    Returns a dictionary of optimal parameters and their associated uncertainties.
    wave: wavelength array
    spec: flux density array
    spec_err: flux density error array
    row: the row of the megatab containing the fitting results to use as priors
    width: the width (in Angstroms) around the Lya peak to use for fitting
    plot_result: whether to plot the fitting result
    """
    # First use a constant baseline with the refit_lya_line function
    fit_const = refit_lya_line(wave, spec, spec_err, row, width=width, baseline='const',
                               plot_result=plot_result, mc_niter=mc_niter, save_plots=save_plots, plot_dir=plot_dir)
    
    # Check the reduced chi-squared of the fit -- if it's good enough, return it
    if fit_const and fit_const[3] < rchsq_thresh:
        print("Constant baseline fit is good enough; returning result.")
        return fit_const
    
    # If not, try a linear baseline
    print("Trying linear baseline fit...")
    fit_lin = refit_lya_line(wave, spec, spec_err, row, width=width, baseline='lin',
                             plot_result=plot_result, mc_niter=mc_niter, save_plots=save_plots, plot_dir=plot_dir)

    if fit_lin and fit_lin[3] < rchsq_thresh:
        print("Linear baseline fit is good enough; returning result.")
        return fit_lin
    
    # If still not good enough, try a damped Lyman alpha baseline
    print("Trying damped Lyman alpha baseline fit...")
    fit_damp = refit_lya_line(wave, spec, spec_err, row, width=width, baseline='damp',
                              plot_result=plot_result, mc_niter=mc_niter, save_plots=save_plots, plot_dir=plot_dir)

    if fit_damp and fit_damp[3] < rchsq_thresh:
        print("Damped Lyman alpha baseline fit is good enough; returning result.")
        return fit_damp
    
    # If none of the fits were good enough, return the one with the lowest reduced chi-squared
    # note that {}, {}, None, None are possible return values if fitting failed
    fits = [fit_const, fit_lin, fit_damp]
    best_fit = min(fits, key=lambda x: x[3] if x[3] else np.inf)
    # Get the fit type for reporting -- bearing in mind that the length of fits may vary
    best_fit_type = ['const', 'lin', 'damp'][fits.index(best_fit)] if best_fit in fits else 'unknown'
    print(f"Returning best fit ({best_fit_type}) with reduced chi-squared = {best_fit[3]:.2f}")
    return best_fit


def get_reduced_chisq(y, ymodel, yerr, nparams):
    """Calculate the reduced chi-squared statistic."""
    residuals = y - ymodel
    chisq = np.nansum((residuals / yerr) ** 2)
    dof = len(y) - nparams
    if dof <= 0:
        return np.inf
    return chisq / dof
    

def refit_lya_line(wave, spec, spec_err, row, width=50, baseline = 'const', plot_result = True,
                   mc_niter=1000, save_plots = False, plot_dir = './'):
    """Refit the Lyman alpha line based on prior fitting results. Always fit a single and
    double peaked profile in parallel and then return the best one based on reduced chi-squared.
    Returns a dictionary of optimal parameters and their associated uncertainties.
    wave: wavelength array
    spec: flux density array
    spec_err: flux density error array
    row: the row of the megatab containing the fitting results to use as priors
    width: the width (in Angstroms) around the Lya peak to use for fitting
    baseline: type of baseline to use ('const', 'lin', or 'damp')
    plot_result: whether to plot the fitting result
    """
    # Get the wavelength of the red Lya peak from the table
    lya_peak = row['LPEAKR']
    
    # First try a double-peaked fit
    print(f"Refitting {row['CLUSTER']} {row['iden']}...")
    
    # Initial guesses from the table, or semi-generic if not available
    amp_r_init = row['AMPR']
    cen_r_init = row['LPEAKR']
    wid_r_init = row['DISPR']
    asy_r_init = row['ASYMR']
    amp_b_init = row['AMPB'] if not np.isnan(row['AMPB']) else 0.1 * row['AMPR']
    cen_b_init = row['LPEAKB'] if not np.isnan(row['LPEAKB']) else row['LPEAKR'] - 5.0
    wid_b_init = row['DISPB'] if not np.isnan(row['DISPB']) else row['DISPR']
    asy_b_init = -1 * row['ASYMR']
    cont_init  = [row['CONT'] if not np.isnan(row['CONT']) else 0.0] # This needs to be a list for appending later

    # Make a list of parameter names for reference in the order used by the model
    param_names = ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB',
                   'AMPR', 'LPEAKR', 'DISPR', 'ASYMR',
                   'CONT']
    
    # Append baseline parameters if needed
    if baseline == 'lin':
        slope_init = row['SLOPE'] if not np.isnan(row['SLOPE']) else 0
        param_names.append('SLOPE')
        cont_init = [*cont_init, slope_init]
        cont_lower = (-50, -np.inf)
        cont_upper = (2000, np.inf)
    elif baseline == 'damp':
        # Get redshift to adjust initial guess for fwhm as needed
        z_init = row['Z'] if not np.isnan(row['Z']) else cen_r_init / 1215.67 - 1
        tau_init = row['TAU'] if not np.isnan(row['TAU']) else 20.0
        # Initial guess for fwhm is 150km/s in rest frame, which we convert to observed frame wavelength
        fwhm_init = (150 / c) * 1215.67 * (1 + z_init)
        # If there is a value in the table, use it instead
        fwhm_init = row['FWHM'] if not np.isnan(row['FWHM']) else fwhm_init
        # Initial guess for LPEAK_ABS is 2.5 Angstroms blueward of the red peak
        lpeak_abs_init = row['LPEAK_ABS'] if not np.isnan(row['LPEAK_ABS']) else cen_r_init - 2.5
        param_names.extend(['TAU', 'FWHM', 'LPEAK_ABS'])
        cont_init = [*cont_init, tau_init, fwhm_init, lpeak_abs_init]
        # Impose bounds of fwhm > 1.25 Angstroms observed frame and less than 500km/s
        cont_lower = (-50 , 10, (100 / c) * 1215.67 * (1 + z_init), lpeak_abs_init - 10)
        cont_upper = (2000, 100, (500 / c) * 1215.67 * (1 + z_init), lpeak_abs_init + 10)
    else:
        cont_lower = (-50,)
        cont_upper = (2000,)

    # Define initial parameters for the double-peaked model
    p0 = [amp_b_init, cen_b_init, wid_b_init, asy_b_init,
            amp_r_init, cen_r_init, wid_r_init, asy_r_init,
            *cont_init]  # baseline
    
    # Define bounds for the parameters
    bounds = (
        [0, cen_b_init - 15, 0.625, -0.5, 
         0, cen_r_init - 5,  0.625, -0.5, 
         *cont_lower],  # lower bounds
        [10000, cen_b_init + 10, 6, 0.1, 
         10000, cen_r_init + 10, 6, 0.5, 
         *cont_upper]   # upper bounds
    )

    # Figure out which function to use based on baseline type
    mdl_func = mdl.lya_dpeak_lin if baseline == 'lin' else \
               mdl.lya_dpeak_damp if baseline == 'damp' else \
               mdl.lya_dpeak

    # Get mask for sky lines and bad values
    fitmask = generate_spec_mask(wave, spec, spec_err,
                                 lya_peak, width, 'LYALPHA')
    
    # Make sure there are enough good points to fit
    if fitmask.sum() < 10:
        print(f"Not enough good points to fit Lya for {row['CLUSTER']} {row['iden']}.")
        return {}, {}, None, None

    best_popt, popt_double, popt_single = None, None, None
    best_perr, perr_double, perr_single = None, None, None
    best_rchsq, rchsq_double, rchsq_single = np.inf, np.inf, np.inf
    best_param_names = None

    # First fit the double-peaked model, trying multiple initial guesses for the blue peak
    for shift in [0, -5, 5, -10, 10]: # Perform five initial fits, moving the blue peak initial guess each time
        p0[1] = cen_b_init + shift
        
        # Make sure the initial guesses are always within bounds
        p0, bounds = check_inputs(p0, bounds)
        try:
            popt, pcov = curve_fit(mdl_func, wave[fitmask], spec[fitmask],
                                   p0=p0, sigma=spec_err[fitmask],
                                   bounds=bounds, absolute_sigma=True,
                                   max_nfev = 100000, method = 'trf')
            perr = np.sqrt(np.diag(pcov))
            if spectro.is_reasonable_dpeak(popt, perr):
                popt_double  = popt
                perr_double  = perr
                rchsq_double = get_reduced_chisq(spec[fitmask], 
                                               mdl_func(wave[fitmask], *popt), 
                                               spec_err[fitmask], 
                                               len(popt))
                break  # Exit the loop if a good fit is found
        except (RuntimeError, ValueError) as e:
            print(f"Fit attempt with blue peak shift {shift} failed: {e}")
            continue
    # if best_popt is not None and best_perr is not None:
    #     print(f"Double-peaked fit successful for {row['CLUSTER']} {row['iden']}.")
    #     # If the fit was successful, populate the parameter dictionary
    #     param_dict = dict(zip(param_names, best_popt))
    #     error_dict = dict(zip(param_names, best_perr))
    if popt_double is None or perr_double is None:
        print(f"No reasonable double-peaked fit found for {row['CLUSTER']} {row['iden']}.")
        print("Moving to single-peaked fit...")

    # Now perform a single-peaked fit (with multiple attempts in case of failure)
    p0_single = [amp_r_init, cen_r_init, wid_r_init, asy_r_init, *cont_init]  # baseline
    bounds_single = (
        [0,     cen_r_init - 5, 1.25, -0.5,  *cont_lower],  # lower bounds
        [10000, cen_r_init + 5, 50,    0.5,  *cont_upper]   # upper bounds
    )
    mdl_func_single = mdl.lya_speak_lin if baseline == 'lin' else \
                mdl.lya_speak_damp if baseline == 'damp' else \
                mdl.lya_speak
    # Ensure initial guesses are within bounds
    p0_single, bounds_single = check_inputs(p0_single, bounds_single)

    for _ in range(3): # Try up to three times
        try:
            popt, pcov = curve_fit(mdl_func_single, wave[fitmask], spec[fitmask],
                                    p0=p0_single, sigma=spec_err[fitmask],
                                    bounds=bounds_single, absolute_sigma=True,
                                    max_nfev = 100000, method = 'trf')
            popt_single  = popt
            perr_single = np.sqrt(np.diag(pcov))
            rchsq_single = get_reduced_chisq(spec[fitmask], 
                                            mdl_func_single(wave[fitmask], *popt), 
                                            spec_err[fitmask], 
                                            len(popt))

            print(f"Single-peaked fit successful for {row['CLUSTER']} {row['iden']}.")

            break  # Exit the loop if fit was successful

        except (RuntimeError, ValueError) as e:
            print(f"Single-peaked fit also failed for {row['CLUSTER']} {row['iden']}: {e}")
        
    # Now compare the single and double peaked fits if both were successful
    if popt_double is not None and perr_double is not None and popt_single is not None and perr_single is not None:
        if rchsq_double < rchsq_single:
            print(f"Double-peaked fit is better (reduced chi-squared = {rchsq_double:.2f}) than single-peaked fit ({rchsq_single:.2f}).")
            best_popt = popt_double
            best_perr = perr_double
            best_rchsq = rchsq_double
            best_param_names = param_names  # Use full parameter names
            best_bounds = bounds
        else:
            print(f"Single-peaked fit is better (reduced chi-squared = {rchsq_single:.2f}) than double-peaked fit ({rchsq_double:.2f}).")
            best_popt = popt_single
            best_perr = perr_single
            best_rchsq = rchsq_single
            # Exclude blue peak parameters from the names
            best_param_names = [n for n in param_names if n[-1] != 'B']
            best_bounds = bounds_single
    elif popt_double is not None and perr_double is not None:
        best_popt = popt_double
        best_perr = perr_double
        best_rchsq = rchsq_double
        best_param_names = param_names  # Use full parameter names
        best_bounds = bounds
    elif popt_single is not None and perr_single is not None:
        best_popt = popt_single
        best_perr = perr_single
        best_rchsq = rchsq_single
        best_param_names = [n for n in param_names if n[-1] != 'B']
        best_bounds = bounds_single
    else:
        print(f"Both single and double-peaked fits failed for {row['CLUSTER']} {row['iden']}.")
        return {}, {}, None, None
    
    # If we got here, we have a best fit. Populate the parameter dictionary
    param_dict = dict(zip(best_param_names, best_popt))
    error_dict = dict(zip(best_param_names, best_perr))

    # Now get the final parameters and errors using bootstrapping
    # Using the LyaProfile class which is in this module
    initial_profile = LyaProfile(param_dict, error_dict)
    # Now use the class method to generate uncertainties
    final_params, final_errors, \
        final_function, final_reduced_chisq \
            = initial_profile.fit_to(wave, spec, spec_err, mask=fitmask,
                                     bounds=best_bounds, use_bootstrap=True,
                                     bootstrap_params={
                                         'niter':           mc_niter,
                                         'autocorrelation': False,
                                         'max_nfev':        2000
                                     })
    
    # Catch cases where the fit failed
    if final_params is None or final_errors is None:
        print(f"Final fitting with bootstrapping failed for {row['CLUSTER']} {row['iden']}.")
        return {}, {}, None, None

    # Add dummy values for the blue peak parameters if needed
    if 'AMPB' not in final_params.keys():
        for pname in param_names[:4]:
            final_params[pname] = np.nan
            final_errors[pname] = np.nan

    # Dictionary of full baseline names for plotting
    basenames = {
        'const': 'Constant',
        'lin': 'Linear',
        'damp': 'Absorption'
    }

    if plot_result and final_function is not None:
        # Plot the fit result, decomposed into lyman alpha and baseline components
        # Make a fine wavelength grid for plotting the model
        finegrid = np.linspace(wave[fitmask].min(), wave[fitmask].max(), 1000)
        # Total model
        final_model = final_function(finegrid, *best_popt)
        # Emission model only (extract the emission parameters, then use either single or double peak function)
        # Appending zero to the end for the continuum level
        emission_popt = best_popt.copy()[:-len(cont_init)]
        rpeak_popt = emission_popt[:4]
        bpeak_model = 0
        if len(emission_popt) == 8: # double peaked
            rpeak_popt = emission_popt[4:8]
            bpeak_popt = emission_popt[:4]
            bpeak_model = mdl.lya_speak(finegrid, *bpeak_popt, 0.0) # Append zero for continuum
        rpeak_model = mdl.lya_speak(finegrid, *rpeak_popt, 0.0) # Append zero for continuum
        # Baseline only -- just subtract the emission model from the total
        baseline_model = final_model - rpeak_model - bpeak_model

        # Now plot
        plt.figure(figsize=(6,3), facecolor='white')
        # Data with error bars
        plt.step(wave[fitmask], spec[fitmask], where='mid', color='black', alpha=0.75, label='Data')
        plt.fill_between(wave[fitmask], spec[fitmask] - spec_err[fitmask], spec[fitmask] + spec_err[fitmask],
                         color='gray', step='mid', alpha=0.3, label='Error')
        # Model components
        # Total model
        plt.plot(finegrid, final_function(finegrid, *best_popt),
                 color='fuchsia', label='Best Fit')
        # Emission only
        plt.plot(finegrid, rpeak_model,
                 color='red', linestyle='--', label='Red Peak')
        if len(emission_popt) == 8: # double peaked
            plt.plot(finegrid, bpeak_model,
                     color='royalblue', linestyle='--', label='Blue Peak')
        # Baseline only
        plt.plot(finegrid, baseline_model,
                 color='tab:green', linestyle=':', label=f'{basenames[baseline]} Baseline')
        plt.xlim(lya_peak - 50, lya_peak + 50)
        plt.xlabel('Wavelength [\AA]')
        plt.ylabel('Flux Density [$10^{-20}$\,erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$]')
        plt.title(f"{row['CLUSTER']} {row['iden']} "+r"Lyman-$\alpha$ Fit")
        plt.legend()
        if save_plots:
            plt.savefig(f"{plot_dir}/{row['CLUSTER']}_{row['iden']}_lya_fit.png", dpi=300)
        plt.show()

    return final_params, final_errors, initial_profile, final_reduced_chisq


def refit_other_line(wave, spec, spec_err, row, line_tab_row, width=25, ax_in=None):
    """Refit a non-Lyman alpha emission line based on prior fitting results.
    Uses a single Gaussian plus linear baseline model with initial guesses from the table row.
    In cases where no significant fit was previously found, gets initial guesses from the R21 catalogue.
    If a secondary line of a doublet is passed, the primary is inferred and passed to the fitting function instead.
    Returns a dictionary of optimal parameters and their associated uncertainties.
    wave:         wavelength array
    spec:         flux density array
    spec_err:     flux density error array
    row:          the row of the megatab containing the fitting results to use as priors
    line_tab_row: the row of the line table containing information about the line to fit
    width:        the width (in Angstroms) around the line to use for fitting
    """
    line_name = line_tab_row['LINE']

    # Was a significant fit found previously?
    significant_fit = row['SNR_'+line_name] > 3.0

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
            primary_line = const.slines[np.where(const.slines == line_name)[0][0] - 1][0]
            secondary_line = line_name
            doublet_ratio = const.wavedict[secondary_line] / const.wavedict[primary_line]
    # What kind of function do we need to use, single or double Gaussian?
    model_func = mdl.gaussian_doublet(doublet_ratio) if doublet else mdl.gaussian

    # Calculate observed wavelength, multiplying by the ratio of the primary line rest wavelength to this line's rest wavelength
    # (this value will be 1.0 if this is the primary line)
    r21_observed_wavelength = line_tab_row['LBDA_OBS'] * const.wavedict[primary_line] / const.wavedict[line_name]

    print(f"Refitting {primary_line} at {r21_observed_wavelength:.2f} Angstroms for {row['CLUSTER']} {row['iden']}...")
    
    # Names of parameters to be fitted
    param_names = ['FLUX', 'LPEAK', 'FWHM']
    
    # Initial guesses from the table, or the R21 catalogue if not available
    flux_init = row[f'FLUX_{primary_line}'] if significant_fit else line_tab_row['FLUX']
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
        flux_init_2 = row[f'FLUX_{secondary_line}'] if significant_fit else flux_init
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

    # Now perform fitting with error handling
    initial_guesses = dict(zip(param_names, p0))
    bounds_dict = {param : (bounds[0][i], bounds[1][i]) for i, param in enumerate(param_names)}
    # Make sure initial guesses are not out of bounds:
    for param, value in initial_guesses.items():
        if value < bounds_dict[param][0]:
            initial_guesses[param] = bounds_dict[param][0] * 1.1 # If the initial guess is smaller than the bounds, replace with this
        elif value > bounds_dict[param][1]:
            initial_guesses[param] = bounds_dict[param][1] * 0.9 # If the initial guess is greater than the bounds, replace with this
    fit = fit_line(wave, spec, spec_err, primary_line, initial_guesses, bounds=bounds_dict,
                         continuum_buffer=width, plot_result=True, ax_in=ax_in)
    
    if 'param_dict' not in fit:
        print(f"Fit failed for {primary_line} in {row['CLUSTER']} {row['iden']}.")
        nandict = {param: np.nan for param in initial_guesses.keys()}
        return nandict, nandict, None, np.nan
    else:
        return fit['param_dict'], fit['error_dict'], fit['model'], fit['reduced_chisq']
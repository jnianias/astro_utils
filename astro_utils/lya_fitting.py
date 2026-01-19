"""
Lyman-alpha specific fitting functions.

This module contains high-level functions for fitting Lyman-alpha emission lines,
including functions that handle baseline selection, model comparison, and refitting.
"""

import numpy as np
import matplotlib.pyplot as plt

from . import constants as const
from . import models as mdl
from . import spectroscopy as spectro
from . import plotting as plot
from .spectroscopy import generate_spec_mask
from .fitting import check_inputs, gen_bounds
from .lya_profile import LyaProfile
from scipy.optimize import curve_fit

w_lya = const.wavedict['LYALPHA']  # Lyman-alpha rest wavelength in Angstroms

def get_reduced_chisq(y, ymodel, yerr, nparams):
    """
    Calculate the reduced chi-squared statistic.
    
    Parameters
    ----------
    y : array-like
        Observed data.
    ymodel : array-like
        Model evaluated at the same points as y.
    yerr : array-like
        Errors associated with the observed data.
    nparams : int
        Number of parameters in the model.
    
    Returns
    -------
    float
        The reduced chi-squared statistic.
    """
    residuals = y - ymodel
    chisq = np.nansum((residuals / yerr) ** 2)
    dof = len(y) - nparams
    if dof <= 0:
        return np.inf
    return chisq / dof

def fit_lya_line(wave, spec, spec_err, initial_guesses, iden, cluster, baseline='auto', width=50,
                 bounds={}, gen_bounds_kwargs={}, plot_result=False, mc_niter=1000, rchsq_thresh=2.0,
                 save_plots=False, plot_dir='./', ax_in=None, spec_type='aper'):
    """
    Master function to fit a Lyman alpha profile to the provided spectrum. Can handle different baseline types,
    and automatically selects between single and double-peaked profiles unless specified otherwise, can use Monte
    Carlo resampling to estimate parameter uncertainties, and can produce diagnostic plots.

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    spec : array-like
        Spectrum array.
    spec_err : array-like
        Spectrum error array.
    initial_guesses : dict or Astropy Table row
        Initial guesses for the fit parameters. If a Table row is provided, it should contain the necessary columns.
    baseline : str, optional
        Baseline type to use ('const', 'lin', 'damp', or 'auto' to try all).
    width : float, optional
        Width around the line center to consider for fitting (default is 50).
    bounds : dict, optional
        Dictionary of bounds for the fitting parameters.
    gen_bounds_kwargs : dict, optional
        Additional keyword arguments for generating bounds.
    plot_result : bool, optional
        Whether to plot the fitting result.
    mc_niter : int, optional
        Number of Monte Carlo iterations for error estimation (default is 1000).
    rchsq_thresh : float, optional
        Reduced chi-squared threshold for accepting a fit when baseline='auto' (default is 2.0).
    save_plots : bool, optional
        Whether to save the plots to disk.
    plot_dir : str, optional
        Directory to save plots if save_plots is True.
    spec_type : str, optional
        Type of spectrum being fitted (for labeling purposes, default: 'aper').
    
    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and baseline type.
        Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 'baseline', 
        'fit_mask', 'wl_fit', 'spec_fit', 'err_fit'.
        Empty dict {} if fit failed.
    """
    if baseline == 'auto':
        # Try constant, then linear, then damped baselines
        fit_result = fit_lya_autobase(wave, spec, spec_err, initial_guesses, iden,
                                        cluster, bounds=bounds, gen_bounds_kwargs=gen_bounds_kwargs, 
                                        width=width, plot_result=plot_result,
                                        mc_niter=mc_niter, rchsq_thresh=rchsq_thresh,
                                        save_plots=save_plots, plot_dir=plot_dir)
        return fit_result
    else:
        # Fit with specified baseline type
        fit_result = fit_lya_line(wave, spec, spec_err, initial_guesses, iden, cluster,
                                    bounds=bounds, gen_bounds_kwargs=gen_bounds_kwargs,
                                    width=width, baseline=baseline,
                                    plot_result=plot_result,
                                    mc_niter=mc_niter,
                                    save_plots=save_plots, plot_dir=plot_dir)
        return fit_result


def fit_lya_autobase(wave, spec, spec_err, initial_guesses, iden, cluster, 
                     width=50, bounds={}, gen_bounds_kwargs={}, plot_result=True, 
                     mc_niter=500, rchsq_thresh=2.0, save_plots = False, 
                     plot_dir = './'):
    """
    Fit the Lyman alpha line based on prior fitting results (if present). Here,
    we compare single and double-peaked fits and return the best one based on reduced chi-squared.
    Baselines are handled in increasing order of complexity (const, lin, damp).
    
    Parameters
    ----------
    wave : array-like
        Wavelength array.
    spec : array-like
        Spectrum array.
    spec_err : array-like
        Spectrum error array.
    row : dict-like
        The row of the megatab containing the fitting results to use as priors.
    width : float, optional
        The width (in Angstroms) around the Lya peak to use for fitting.
    bounds : dict, optional
        Dictionary of bounds for the fitting parameters.
    gen_bounds_kwargs : dict, optional
        Additional keyword arguments for generating bounds.
    plot_result : bool, optional
        Whether to plot the fitting result.
    mc_niter : int, optional
        Number of Monte Carlo iterations for error estimation.
    rchsq_thresh : float, optional
        Reduced chi-squared threshold for accepting a fit.
    save_plots : bool, optional
        Whether to save the plots to disk.
    plot_dir : str, optional
        Directory to save plots if save_plots is True.

    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and baseline type.
        Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 'baseline', 
        'fit_mask', 'wl_fit', 'spec_fit', 'err_fit'.
        Empty dict {} if all fits failed.
    """
    # First use a constant baseline with the refit_lya_line function
    fit_const = fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster, width=width, baseline='const',
                               plot_result=plot_result, mc_niter=mc_niter, save_plots=save_plots, plot_dir=plot_dir)
    
    # Check the reduced chi-squared of the fit -- if it's good enough, return it
    if fit_const and fit_const.get('reduced_chisq') and fit_const['reduced_chisq'] < rchsq_thresh:
        print("Constant baseline fit is good enough; returning result.")
        return fit_const
    
    # If not, try a linear baseline
    print("Trying linear baseline fit...")
    fit_lin = fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster, width=width, baseline='lin',
                             plot_result=plot_result, mc_niter=mc_niter, save_plots=save_plots, plot_dir=plot_dir)

    if fit_lin and fit_lin.get('reduced_chisq') and fit_lin['reduced_chisq'] < rchsq_thresh:
        print("Linear baseline fit is good enough; returning result.")
        return fit_lin
    
    # If still not good enough, try a damped Lyman alpha baseline
    print("Trying damped Lyman alpha baseline fit...")
    fit_damp = fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster, width=width, baseline='damp',
                              plot_result=plot_result, mc_niter=mc_niter, save_plots=save_plots, plot_dir=plot_dir)

    if fit_damp and fit_damp.get('reduced_chisq') and fit_damp['reduced_chisq'] < rchsq_thresh:
        print("Damped Lyman alpha baseline fit is good enough; returning result.")
        return fit_damp
    
    # If none of the fits were good enough, return the one with the lowest reduced chi-squared
    fits = [fit_const, fit_lin, fit_damp]
    best_fit = min(fits, key=lambda x: x.get('reduced_chisq', np.inf))
    # Get the fit type for reporting
    best_fit_type = ['const', 'lin', 'damp'][fits.index(best_fit)] if best_fit in fits else 'unknown'
    rchsq_value = best_fit.get('reduced_chisq', np.nan)
    print(f"Returning best fit ({best_fit_type}) with reduced chi-squared = {rchsq_value:.2f}")
    return best_fit

def fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster, 
            bounds={}, gen_bounds_kwargs={}, width=50, baseline='const', 
            plot_result=True, mc_niter=1000, save_plots=False, 
            plot_dir='./'):
    """
    Refit the Lyman alpha line based on prior fitting results. Here, we compare single and double-peaked fits
    and return the best one based on reduced chi-squared.
    
    Parameters
    ----------
    wave : array-like
        Wavelength array.
    spec : array-like
        Spectrum array.
    spec_err : array-like
        Spectrum error array.
    initial_guesses : dict-like
        Dictionary or table row containing initial guesses for the fitting parameters.
    iden : str
        Identifier for the source being fitted.
    cluster : str
        Cluster name for the source being fitted.
    bounds : dict, optional
        Dictionary of bounds for the fitting parameters.
    gen_bounds_kwargs : dict, optional
        Additional keyword arguments for generating bounds.
    width : float, optional
        The width (in Angstroms) around the Lya peak to use for fitting.
    baseline : str, optional
        Baseline type ('const', 'lin', 'damp').
    plot_result : bool, optional
        Whether to plot the fitting result.
    mc_niter : int, optional
        Number of Monte Carlo iterations for error estimation.
    save_plots : bool, optional
        Whether to save the plots to disk.
    plot_dir : str, optional
        Directory to save plots if save_plots is True.
    
    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and baseline type.
        Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 'baseline', 
        'fit_mask', 'wl_fit', 'spec_fit', 'err_fit', 'popt', 'pcov', 'method'.
        Empty dict {} if fit failed.
    """

    # Convert initial_guesses to a dict if it's an Astropy Table row
    if not isinstance(initial_guesses, dict):
        initial_guesses = {col: initial_guesses[col] for col in initial_guesses.colnames if not np.isnan(initial_guesses[col])}
    else: # remove any NaN entries
        initial_guesses = {k: v for k, v in initial_guesses.items() if not np.isnan(v)}


    # First try a double-peaked fit
    print(f"\nFitting Lyman-α line to {cluster} {iden}...")

    # Get the wavelength of the red Lya peak from the table
    lya_peak = initial_guesses['LPEAKR']
    if np.isnan(lya_peak):
        raise ValueError("\nInitial guess for LPEAKR is NaN; cannot proceed with fitting.")
    
    # Initial guesses from the table, or semi-generic if not available
    try:
        amp_r_init = initial_guesses['AMPR'] #These will throw an error if not present, which is the desired behavior
        cen_r_init = initial_guesses['LPEAKR']
        wid_r_init = initial_guesses['DISPR']
        asy_r_init = initial_guesses['ASYMR']
    except KeyError as e:
        raise KeyError(f"\nMissing required initial guess parameter: {e}. Cannot proceed with fitting.")

    amp_b_init = initial_guesses.get('AMPB', 0.1 * initial_guesses['AMPR'])
    cen_b_init = initial_guesses.get('LPEAKB', initial_guesses['LPEAKR'] - 5.0)
    wid_b_init = initial_guesses.get('DISPB', initial_guesses['DISPR'])
    asy_b_init = -1 * initial_guesses['ASYMR']
    cont_init  = [initial_guesses.get('CONT', 0.0)] # This needs to be a list for appending later
    
    # Speed of light constant
    c = const.c

    # Make a list of parameter names for reference in the order used by the model
    param_names = ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB',
                   'AMPR', 'LPEAKR', 'DISPR', 'ASYMR',
                   'CONT']
    
    # Append baseline parameters if needed
    if baseline == 'lin':
        slope_init = initial_guesses.get('SLOPE', 0.0)
        param_names.append('SLOPE')
        cont_init = [*cont_init, slope_init]
        # cont_lower = (-50, -np.inf) # The bound on cont is chosen to allow only small negative values
        # cont_upper = (2000, np.inf) # The bound on cont is chosen to allow reasonable positive values
    elif baseline == 'damp':
        # Get redshift to adjust initial guess for fwhm as needed
        z_init = initial_guesses['Z'] if not np.isnan(initial_guesses['Z']) else cen_r_init / 1215.67 - 1
        tau_init = initial_guesses['TAU'] if not np.isnan(initial_guesses['TAU']) else 20.0
        # Initial guess for fwhm is 150km/s in rest frame, which we convert to observed frame wavelength
        fwhm_init = (150 / c) * 1215.67 * (1 + z_init)
        # If there is a value in the table, use it instead
        fwhm_init = initial_guesses['FWHM'] if not np.isnan(initial_guesses['FWHM']) else fwhm_init
        # Initial guess for LPEAK_ABS is 2.5 Angstroms blueward of the red peak
        lpeak_abs_init = initial_guesses['LPEAK_ABS'] if not np.isnan(initial_guesses['LPEAK_ABS']) else cen_r_init - 2.5
        param_names.extend(['TAU', 'FWHM', 'LPEAK_ABS'])
        cont_init = [*cont_init, tau_init, fwhm_init, lpeak_abs_init]
        # Impose bounds of fwhm > 1.25 Angstroms observed frame and less than 500km/s
        # cont_lower = (-50 , 10, (100 / c) * 1215.67 * (1 + z_init), lpeak_abs_init - 10)
        # cont_upper = (2000, 100, (500 / c) * 1215.67 * (1 + z_init), lpeak_abs_init + 10)
    # else:
    #     cont_lower = (-50,)
    #     cont_upper = (2000,)

    # Define initial parameters for the double-peaked model
    p0 = [amp_b_init, cen_b_init, wid_b_init, asy_b_init,
            amp_r_init, cen_r_init, wid_r_init, asy_r_init,
            *cont_init]  # baseline
    
    # Define bounds for the parameters
    bounds = gen_bounds(initial_guesses, 'LYALPHA')
    # bounds = (
    #     [0, cen_b_init - 15, 0.625, -0.5, 
    #      0, cen_r_init - 5,  0.625, -0.5, 
    #      *cont_lower],  # lower bounds
    #     [10000, cen_b_init + 10, 6, 0.1, 
    #      10000, cen_r_init + 10, 6, 0.5, 
    #      *cont_upper]   # upper bounds
    # )

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
        return {}

    best_popt, popt_double, popt_single = None, None, None
    best_perr, perr_double, perr_single = None, None, None
    best_pcov, pcov_double, pcov_single = None, None, None
    best_rchsq, rchsq_double, rchsq_single = np.inf, np.inf, np.inf
    best_param_names = None
    best_method = None

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
                pcov_double  = pcov
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
            pcov_single = pcov
            rchsq_single = get_reduced_chisq(spec[fitmask], 
                                            mdl_func_single(wave[fitmask], *popt), 
                                            spec_err[fitmask], 
                                            len(popt))

            print(f"Single-peaked fit successful for {cluster} {iden}.")

            break  # Exit the loop if fit was successful

        except (RuntimeError, ValueError) as e:
            print(f"Single-peaked fit also failed for {cluster} {iden}: {e}")
        
    # Now compare the single and double peaked fits if both were successful
    if popt_double is not None and perr_double is not None and popt_single is not None and perr_single is not None:
        if rchsq_double < rchsq_single:
            print(f"Double-peaked fit is better (reduced chi-squared = {rchsq_double:.2f}) than single-peaked fit ({rchsq_single:.2f}).")
            best_popt = popt_double
            best_perr = perr_double
            best_pcov = pcov_double
            best_rchsq = rchsq_double
            best_param_names = param_names  # Use full parameter names
            best_bounds = bounds
            best_method = 'double-peaked'
        else:
            print(f"Single-peaked fit is better (reduced chi-squared = {rchsq_single:.2f}) than double-peaked fit ({rchsq_double:.2f}).")
            best_popt = popt_single
            best_perr = perr_single
            best_pcov = pcov_single
            best_rchsq = rchsq_single
            # Exclude blue peak parameters from the names
            best_param_names = [n for n in param_names if n[-1] != 'B']
            best_bounds = bounds_single
            best_method = 'single-peaked'
    elif popt_double is not None and perr_double is not None:
        best_popt = popt_double
        best_perr = perr_double
        best_pcov = pcov_double
        best_rchsq = rchsq_double
        best_param_names = param_names  # Use full parameter names
        best_bounds = bounds
        best_method = 'double-peaked'
    elif popt_single is not None and perr_single is not None:
        best_popt = popt_single
        best_perr = perr_single
        best_pcov = pcov_single
        best_rchsq = rchsq_single
        best_param_names = [n for n in param_names if n[-1] != 'B']
        best_bounds = bounds_single
        best_method = 'single-peaked'
    else:
        print(f"Both single and double-peaked fits failed for {row['CLUSTER']} {row['iden']}.")
        return {}
    
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
        print(f"Final fitting with bootstrapping failed for {cluster} {iden}.")
        return {}

    # Add dummy values for the blue peak parameters if needed
    if 'AMPB' not in final_params.keys():
        for pname in param_names[:4]:
            final_params[pname] = np.nan
            final_errors[pname] = np.nan
    
    # Build the fit_result dictionary to return
    fit_result = {
        'param_dict': final_params,
        'error_dict': final_errors,
        'model': final_function,
        'reduced_chisq': final_reduced_chisq,
        'baseline': baseline,
        'fit_mask': fitmask,
        'wl_fit': wave[fitmask],
        'spec_fit': spec[fitmask],
        'err_fit': spec_err[fitmask],
        'popt': best_popt,
        'pcov': best_pcov,
        'method': best_method
    }

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

    return fit_result


# def fit_lya_line_old(wave, spec, spec_err, initial_guesses, bounds='auto', width=50,
#                  plot_result=False, save_plots=False, plot_dir='./', ax_in=None, spec_type='aper'):
#     """
#     Fit the Lyman alpha line with a constant baseline using specified initial parameters.

#     Parameters
#     ----------
#     wave : array-like
#         Wavelength array.
#     spec : array-like
#         Spectrum array.
#     spec_err : array-like
#         Spectrum error array.
#     initial_guesses : dict
#         Dictionary of initial guesses for the fit parameters.
#     bounds : tuple or 'auto', optional
#         Bounds for parameters. If 'auto', default bounds are used based on initial guesses.
#     width : float, optional
#         Width around the line center to consider for fitting (default is 50).
#     spec_type : str, optional
#         Type of spectrum being fitted (for labeling purposes, default: 'aper').

#     Returns
#     -------
#     fit_result : dict
#         Dictionary containing fit parameters, errors, model, and reduced chi-squared.
#         Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 
#         'fit_mask', 'wl_fit', 'spec_fit', 'err_fit', 'popt', 'pcov'.
#         Empty dict {} if fit failed.
#     """
#     # Figure out which function to use based on baseline type
#     mdl_func = mdl.lya_dpeak
    
#     # Extract initial guesses from the dictionary first for mask generation
#     lpeak_init = initial_guesses.get('LPEAKR', initial_guesses.get('LPEAK', 1216))
    
#     # Get mask for sky lines and bad values
#     fitmask = generate_spec_mask(wave, spec, spec_err,
#                                  lpeak_init, width, 'LYALPHA')
#     # Make sure there are enough good points to fit
#     if fitmask.sum() < 10:
#         print("Not enough good points to fit Lya.")
#         return {}
    
#     # Extract initial guesses from the dictionary
#     p0 = [
#         initial_guesses.get('AMPB', 0.1 * initial_guesses['AMPR']),
#         initial_guesses.get('LPEAKB', initial_guesses['LPEAKR'] - 5.0),
#         initial_guesses.get('DISPB', initial_guesses['DISPR']),
#         initial_guesses.get('ASYMB', -1 * initial_guesses['ASYMR']),
#         initial_guesses['AMPR'],
#         initial_guesses['LPEAKR'],
#         initial_guesses['DISPR'],
#         initial_guesses['ASYMR'],
#         initial_guesses.get('CONT', 0.0)
#     ]

#     # Make a list of parameter names for reference in the order used by the model
#     param_names = ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB',
#                    'AMPR', 'LPEAKR', 'DISPR', 'ASYMR',
#                    'CONT']
    
#     # Define default bounds if 'auto' is specified
#     if bounds == 'auto':
#         bounds = (
#             [0, p0[1] - 15, 0.625, -0.5, 
#              0, p0[5] - 5,  0.625, -0.5, 
#              -50],  # lower bounds
#             [10000, p0[1] + 10, 6, 0.1, 
#              10000, p0[5] + 10, 6, 0.5, 
#              2000]   # upper bounds
#         )

#     # Make sure that initial guesses are within bounds
#     p0, bounds = check_inputs(p0, bounds)

#     # Initialise fit results dictionaries
#     fit_results = {k: np.nan for k in initial_guesses.keys()}
#     err_results = {k: np.nan for k in initial_guesses.keys()}
    
#     # Try double-peaked fit first with a few attempts at different initial guesses
#     popt_double, perr_double, pcov_double, rchsq_double = None, None, None, np.inf
#     best_mdl_func = mdl_func

#     for shift in [0, -5, 5, -10, 10]: # Perform five initial fits, moving the blue peak initial guess each time
#         p0[1] = p0[1] + shift
        
#         # Make sure the initial guesses are always within bounds
#         p0, bounds = check_inputs(p0, bounds)
#         try:
#             popt, pcov = curve_fit(mdl_func, wave[fitmask], spec[fitmask],
#                                    p0=p0, sigma=spec_err[fitmask],
#                                    bounds=bounds, absolute_sigma=True,
#                                    max_nfev = 100000, method = 'trf')
#             perr = np.sqrt(np.diag(pcov))
#             if spectro.is_reasonable_dpeak(popt, perr):
#                 popt_double  = popt
#                 perr_double  = perr
#                 pcov_double  = pcov
#                 rchsq_double = get_reduced_chisq(spec[fitmask], 
#                                                mdl_func(wave[fitmask], *popt), 
#                                                spec_err[fitmask], 
#                                                len(popt))
#                 break  # Exit the loop if a good fit is found
#         except (RuntimeError, ValueError) as e:
#             print(f"Fit attempt with blue peak shift {shift} failed: {e}")
#             continue
    
#     # If a successful fit was found, fill the fit_results and err_results dictionaries, 
#     # otherwise move to single-peaked fit
#     if popt_double is not None and perr_double is not None:
#         for i, name in enumerate(param_names):
#             fit_results[name] = popt_double[i]
#             err_results[name] = perr_double[i]
#         print("Double-peaked fit successful.")
        
#         # If plotting is requested, do it here
#         if plot_result:
#             plot.plot_lya_fit(wave[fitmask], spec[fitmask], spec_err[fitmask], popt_double, best_mdl_func,
#                              save_plots=save_plots, plot_dir=plot_dir, ax_in=ax_in, spec_type=spec_type)

#         # Build and return fit_result dictionary
#         return {
#             'param_dict': fit_results,
#             'error_dict': err_results,
#             'model': mdl_func,
#             'reduced_chisq': rchsq_double,
#             'fit_mask': fitmask,
#             'wl_fit': wave[fitmask],
#             'spec_fit': spec[fitmask],
#             'err_fit': spec_err[fitmask],
#             'popt': popt_double,
#             'pcov': pcov_double,
#             'method': 'double-peaked'
#         }
    
#     print("Double-peaked fit failed; trying single-peaked fit...")

#     # Now perform a single-peaked fit
#     p0_single = [p0[4], p0[5], p0[6], p0[7], p0[8]]  # baseline
#     bounds_single = (
#         [0,     p0[5] - 5, 1.25, -0.5,  bounds[0][8]],  # lower bounds
#         [10000, p0[5] + 5, 50,    0.5,  bounds[1][8]]   # upper bounds
#     )

#     mdl_func_single = mdl.lya_speak
    
#     # Ensure initial guesses are within bounds
#     p0_single, bounds_single = check_inputs(p0_single, bounds_single)

#     try:
#         popt, pcov = curve_fit(mdl_func_single, wave[fitmask], spec[fitmask],
#                                 p0=p0_single, sigma=spec_err[fitmask],
#                                 bounds=bounds_single, absolute_sigma=True,
#                                 max_nfev = 100000, method = 'trf')
#         popt_single  = popt
#         perr_single = np.sqrt(np.diag(pcov))
#         pcov_single = pcov
#         rchsq_single = get_reduced_chisq(spec[fitmask], 
#                                         mdl_func_single(wave[fitmask], *popt), 
#                                         spec_err[fitmask], 
#                                         len(popt))

#         for i, name in enumerate(param_names[4:]):  # only the single-peak params
#             fit_results[name] = popt_single[i]
#             err_results[name] = perr_single[i]
#         print("Single-peaked fit successful.")

#         # If plotting is requested, do it here
#         if plot_result:
#             plot.plot_lya_fit(wave[fitmask], spec[fitmask], spec_err[fitmask], popt_single, mdl_func_single,
#                              save_plots=save_plots, plot_dir=plot_dir, ax_in=ax_in, spec_type=spec_type)
        
#         # Build and return fit_result dictionary
#         return {
#             'param_dict': fit_results,
#             'error_dict': err_results,
#             'model': mdl_func_single,
#             'reduced_chisq': rchsq_single,
#             'fit_mask': fitmask,
#             'wl_fit': wave[fitmask],
#             'spec_fit': spec[fitmask],
#             'err_fit': spec_err[fitmask],
#             'popt': popt_single,
#             'pcov': pcov_single,
#             'method': 'single-peaked'
#         }
#     except (RuntimeError, ValueError) as e:
#         print(f"Single-peaked fit also failed: {e}")
#         return {}
    
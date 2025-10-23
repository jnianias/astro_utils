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
from .spectroscopy import generate_spec_mask
from .fitting import check_inputs
from .lya_profile import LyaProfile
from scipy.optimize import curve_fit


def get_reduced_chisq(y, ymodel, yerr, nparams):
    """Calculate the reduced chi-squared statistic."""
    residuals = y - ymodel
    chisq = np.nansum((residuals / yerr) ** 2)
    dof = len(y) - nparams
    if dof <= 0:
        return np.inf
    return chisq / dof


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
    
    # Speed of light constant
    c = const.c

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

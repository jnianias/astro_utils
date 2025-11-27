# Plotting functions for spectral lines

import numpy as np
import matplotlib.pyplot as plt
from . import spectroscopy as spectro
from . import models
import copy

def plotline(iden, clus, idfrom, wln, ax_in, spec_source = '2fwhm', width=100, model=None, title = False,
             hline = None, hspan = None, vline = None, vspan = None,
             # plot_sky = 'red', plot_cluslines = 'magenta', plot_bkg = 'goldenrod', plot_clusspec = 'seagreen', 
             set_ylim = 'mpl', normalise=False, label = None, return_spectrum = False, 
             plotcolor='slateblue', modcolor='maroon'):
    """ Plot a spectrum from the R21 catalog or 2FWHM catalog with various options.
    iden:       ID of the source in the catalog
    clus:       Cluster name (e.g., 'A2744')
    wln:        Central wavelength to plot around
    ax_in:      Matplotlib axis to plot on
    spec_source: 'R21' for R21 catalog, '2fwhm' for 2FWHM catalog
    width:      Wavelength range around wln to plot (total width = 2*width)
    model:      Tuple of (function, params...) to overplot a model fit
    title:      Title for the plot (if False, no title)
    hline:      Horizontal line value to plot (if None, no line)
    hspan:      Horizontal span to plot (if None, no span)
    vline:      Vertical line value to plot (if None, no line)
    vspan:      Vertical span to plot (if None, no span)
    # plot_sky:   Color to plot sky spectrum (if None, no sky)
    # plot_cluslines: Color to plot cluster member lines (if None, no lines)
    # plot_bkg:   Color to plot background spectrum (if None, no background)
    # plot_clusspec: Color to plot cluster spectrum (if None, no cluster spectrum)
    set_ylim:  Y-axis limits (if 'mpl', use Matplotlib defaults, if 'manual', set from data)
    normalise:  Whether to normalise the spectrum
    label:      Label for the spectrum (for legend)
    return_spectrum: Whether to return the plotted spectrum data
    plotcolor:  Color for the main spectrum plot
    modcolor:   Color for the model fit plot
    """

    ax = ax_in

    # Find spectrum file and load it as a table
    load_method = spectro.load_r21_spec if spec_source == 'R21' else spectro.load_aper_spec
    spectab = load_method(clus, iden, idfrom, 'weight_skysub' if spec_source == 'R21' else '2fwhm')

    if spectab is None:
        print(f'No spectrum found for {clus} ID{iden} in {spec_source} catalog.')
        return None

    if normalise: # normalise the spectrum if requested
        normfac = copy.deepcopy(np.abs(np.nanmedian(spectab['spec'].data)))
        spectab['spec'] /= normfac
        spectab['spec_err'] /= normfac

    # region to plot
    region = np.logical_and(wln - width < spectab['wave'], spectab['wave'] < wln + width) 
    if np.sum(region) < 2:
        print(f'No data in the specified wavelength range for {clus} ID{iden}.')
        return None

    # Plot the spectrum
    ax.plot(spectab['wave'][region], spectab['spec'][region], drawstyle='steps-mid', color=plotcolor, 
            alpha=0.8, label=label)
    ax.fill_between(spectab['wave'][region], spectab['spec'][region] - spectab['spec_err'][region],
                spectab['spec'][region] + spectab['spec_err'][region], edgecolor='none',
                facecolor=plotcolor, alpha=0.3, step='mid')
    
    # Axis labels and title
    ax.set_ylabel(r'f$_{\lambda}$ ($10^{-20}$\,erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$)')
    ax.set_xlabel(r'$\lambda$ (\AA)')
    if isinstance(title, str):
        ax.set_title(f'{clus}  ID{iden}' + ' ' + title)

    # Add lines/spans if requested
    if hline:
        ax.axhline(hline, linestyle = '--', alpha=0.5, color='gray')
    if hspan:
        ax.axhspan(hspan[0], hspan[1], alpha=0.3, color='gray')
    if vline:
        for l in vline:
            ax.axvline(l, linestyle = '--', alpha=0.5, color='green')
    if vspan:
        ax.axvspan(vspan[0], vspan[1], alpha=0.3, color='orange')

    # Overplot model if provided
    if model is not None:
        func = model[0]
        highreswl = np.arange(np.nanmin(spectab['wave'][region]), np.nanmax(spectab['wave'][region]), 0.125)
        modplot = func(highreswl, *model[1:])
        if normalise:
            modplot /= normfac
        ax.plot(highreswl, modplot, linestyle = '--', color=modcolor, alpha=0.3, label='fit')
    
    # # Plot cluster member lines
    # if plot_cluslines:
    #     for ln, wl in cluster_lines[row['CLUSTER']].items():
    #         wlz = wl * (cluster_zs[row['CLUSTER']] + 1)
    #         if np.logical_and(spectab['wave'][region][0] < wlz, wlz < spectab['wave'][region][-1]):
    #             ax.axvline(wlz, linestyle=':', color='magenta')
    # # Plot the background spectrum
    # if plot_bkg:
    #     bkgspec = tb(fits.open(f'{clus}_background_spectrum.fits')[1].data)[region]
    #     ax.plot(bkgspec['wavelength'], 1000 * bkgspec['sky_spec'], color=plot_bkg, alpha=0.5, drawstyle='steps-mid')
    # # Plot the cluster spectrum
    # if plot_clusspec:
    #     clusspec = tb(fits.open(f'{clus}_cluster_spectrum.fits')[1].data)[region]
    #     clusspec_norm = clusspec['sky_spec'] * np.nanmedian(spectab['spec'][region]) / np.nanmedian(clusspec['sky_spec'])
    #     ax.plot(clusspec['wavelength'], clusspec_norm, color=plot_clusspec, alpha=0.5, drawstyle='steps-mid')
    # # Plot the sky spectrum
    # if plot_sky:
    #     skytab = get_skytab([spectab['wave'][0], spectab['wave'][-1]])
    #     sky = np.histogram(skytab['lambda'].data,
    #                            bins = np.size(spectab['wave'].data),
    #                            weights = skytab['flux'].data)[0]
    #     ax.step(spectab['wave'][region], sky[region], where='mid', color=plot_sky, alpha = 0.2)
    if set_ylim == 'manual':
        ax.set_ylim(np.min([0., np.nanmin(spectab['spec'][region])]), np.max(spectab['spec'][region]) * 1.1)

    # Legend
    if label or model is not None:
        ax.legend()

    if return_spectrum:
        return spectab[region]


def lya_mod_plot(row, axin, eml=False):
    """
    Plot Lyman alpha models on a given axis in velocity space.
    
    This function generates a high-resolution model of the Lyman alpha line profile
    and plots it in velocity space on the provided axis. It can handle both single-peak
    and double-peak Lyman alpha models with different baseline types (constant, linear,
    or damped) depending on the fit parameters available in the row.
    
    Parameters
    ----------
    row : dict or table row
        A dictionary or table row containing the fit parameters for the Lyman alpha profile.
        Must contain the following keys:
        - 'LPEAKR': Rest-frame wavelength of the red peak (Angstroms)
        - 'AMPR': Amplitude of the red peak
        - 'DISPR': Dispersion of the red peak 
        - 'ASYMR': Asymmetry parameter of the red peak
        - 'CONT': Continuum level
        - 'SNRB': Signal-to-noise ratio of the blue peak
        If SNRB > 3.0, also requires:
        - 'AMPB': Amplitude of the blue peak
        - 'LPEAKB': Rest-frame wavelength of the blue peak (Angstroms)
        - 'DISPB': Dispersion of the blue peak
        - 'ASYMB': Asymmetry parameter of the blue peak
        For linear baseline (if 'SLOPE' is not NaN):
        - 'SLOPE': Linear slope parameter
        For damped baseline (if 'TAU' is not NaN):
        - 'TAU': Damping parameter
        - 'FWHM': Full width at half maximum for damped profile
        - 'LPEAK_ABS': Peak wavelength of absorption component
        If eml=True, also requires:
        - 'DELTAV_LYA': Velocity offset for emission lines (km/s)
    
    axin : matplotlib.axes.Axes
        The matplotlib axis object on which to plot the model.
    
    eml : bool, optional
        If True, applies a velocity offset (DELTAV_LYA) to the model.
        Default is False.
    
    Returns
    -------
    None
        The function plots directly on the provided axis and does not return anything.
    
    Notes
    -----
    - The function creates a high-resolution wavelength grid spanning ±40 Angstroms
      around the red peak wavelength with 1000 points.
    - Model selection hierarchy:
      1. If 'SLOPE' is not NaN: uses linear baseline models (lya_speak_lin/lya_dpeak_lin)
      2. Elif 'TAU' is not NaN: uses damped baseline models (lya_speak_damp/lya_dpeak_damp)  
      3. Else: uses constant baseline models (lya_speak/lya_dpeak)
    - Uses single-peak model if blue SNR <= 3.0, otherwise uses double-peak model.
    - Converts wavelength to velocity using the Lyman alpha rest wavelength (1215.67 Å).
    - The flux is converted from wavelength space to velocity space using dλ/dv.
    - The model is plotted as a dashed maroon line with 60% opacity.
    """
    # Create high-resolution wavelength grid around the red peak
    hireswl = np.linspace(row['LPEAKR'] - 40, row['LPEAKR'] + 40, 1000)
    
    # Determine baseline type (matching logic in lya_profile.py)
    # Check for SLOPE first (linear baseline), but TAU takes precedence if present
    baseline_type = 'const'  # default
    if not np.isnan(row['SLOPE']):
        baseline_type = 'lin'
    elif not np.isnan(row['TAU']):
        baseline_type = 'damp'  # damped overrides linear if both present
    
    if baseline_type == 'lin':
        # Use linear baseline models
        if row['SNRB'] > 3.0:
            # Double-peak with linear baseline
            hiresmod = models.lya_dpeak_lin(hireswl, row['AMPB'], row['LPEAKB'], row['DISPB'], row['ASYMB'],
                                           row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                           row['CONT'], row['SLOPE'])
        else:
            # Single-peak with linear baseline
            hiresmod = models.lya_speak_lin(hireswl, row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                           row['CONT'], row['SLOPE'])
    elif baseline_type == 'damp':
        # Use damped baseline models
        if row['SNRB'] > 3.0:
            # Double-peak with damped baseline
            hiresmod = models.lya_dpeak_damp(hireswl, row['AMPB'], row['LPEAKB'], row['DISPB'], row['ASYMB'],
                                            row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                            row['CONT'], row['TAU'], row['FWHM'], row['LPEAK_ABS'])
        else:
            # Single-peak with damped baseline
            hiresmod = models.lya_speak_damp(hireswl, row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                            row['CONT'], row['TAU'], row['FWHM'], row['LPEAK_ABS'])
    else:
        # Use constant baseline models (original behavior)
        if row['SNRB'] > 3.0:
            # Double-peak with constant baseline
            hiresmod = models.lya_dpeak(hireswl, row['AMPB'], row['LPEAKB'], row['DISPB'], row['ASYMB'],
                                       row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], row['CONT'])
        else:
            # Single-peak with constant baseline
            hiresmod = models.lya_speak(hireswl, row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], row['CONT'])
    
    # Convert wavelength to velocity
    hiresvel = spectro.wave2vel(hireswl, 1215.67, redshift=row['LPEAKR'] / 1215.67 - 1)
    
    # Apply velocity offset if requested
    if eml and 'DELTAV_LYA' in row:
        hiresvel += row['DELTAV_LYA']
    
    # Convert flux from wavelength space to velocity space (dλ/dv)
    dldv = np.ediff1d(hireswl, to_end=np.ediff1d(hireswl)[-1]) / np.ediff1d(hiresvel, to_end=np.ediff1d(hiresvel)[-1])
    
    # Plot the model
    axin.plot(hiresvel[1:-1], dldv[1:-1] * hiresmod[1:-1], 
              color='maroon', alpha=0.6, label=r"model", linestyle='--')
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Column, vstack
from scipy.optimize import curve_fit
from scipy import odr
from scipy.interpolate import interp1d
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.convolution import convolve_fft, Box1DKernel
import warnings
import glob
import os
import astropy.table as aptb
import error_propagation as ep
from mpdaf.MUSE import LSF
from pathlib import Path

# Import constants
from .constants import c, wavedict, doublets, skylines
from . import models as mdl

# Import dictionaries for convenience
skylinedict = skylines
doubletdict = doublets

lsf = LSF(typ='qsim_v1')


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

def get_lsf_fwhm(lsf_lpeak, step = 1.25):
    x0up = np.argwhere(lsf_lpeak > 0.5 * np.max(lsf_lpeak))[0][0]
    y0up = lsf_lpeak[x0up]
    x0down = x0up - 1
    y0down = lsf_lpeak[x0down]
    x1up = np.argwhere(lsf_lpeak > 0.5 * np.max(lsf_lpeak))[-1][0]
    y1up = lsf_lpeak[x1up]
    x1down = x1up + 1
    y1down = lsf_lpeak[x1down]
    x0 = np.interp(0.5 * np.max(lsf_lpeak), np.array([y0down, y0up]), np.array([x0down, x0up]))
    x1 = np.interp(0.5 * np.max(lsf_lpeak), np.array([y1down, y1up]), np.array([x1down, x1up]))
    return (x1 - x0) * step

def muse_lsf_fwhm_poly(lam):
    """Return the FWHM of the MUSE LSF at wavelength lam (in Angstroms).
    Uses the polynomial model given in pyplatefit documentation: 
    https://pyplatefit.readthedocs.io/en/latest/tutorial.html
    """
    return 5.19939 - 7.56746 * 1e-4 * lam + 4.93397 * 1e-8 * lam**2.


#alternative version of wave2vel that uses relativistic relative velocity formula:
def wave2vel(observed_wavelength, rest_wavelength, redshift = 0):
    """
    Convert observed wavelength to velocity in the target object's rest frame,
    accounting for systemic redshift and using the full relativistic Doppler formula.

    Parameters:
    observed_wavelength (float): Observed wavelength in the same units as rest_wavelength.
    rest_wavelength (float): Rest wavelength of the spectral line.
    redshift (float): Systemic redshift of the source.

    Returns:
    float: Velocity in the target object's rest frame, in km/s.
    """
    # Speed of light in km/s
    c = 299792.458

    # Calculate the velocity in the observer's frame (Earth's frame)
    wavelength_ratio = observed_wavelength / rest_wavelength
    beta_obs = (wavelength_ratio**2 - 1) / (wavelength_ratio**2 + 1)
    v_obs = beta_obs * c

    # Calculate the systemic velocity corresponding to the redshift
    beta_sys = ((1 + redshift)**2 - 1) / ((1 + redshift)**2 + 1)
    v_sys = beta_sys * c

    # Use the relativistic velocity addition formula to get the velocity in the target's frame
    v_target = (v_obs - v_sys) / (1 - (v_obs * v_sys) / c**2)

    return v_target

def vel2wave(vel, restLambda, z = 0.):
    """
    Convert velocity in the target object's rest frame to observed wavelength,
    accounting for systemic redshift and using the full relativistic Doppler formula.

    Parameters:
    vel (float): Velocity in the target object's rest frame, in km/s.
    restLambda (float): Rest wavelength of the spectral line.
    z (float): Systemic redshift of the source.

    Returns:
    float: Observed wavelength in the same units as restLambda.
    """
    # Speed of light in km/s
    c = 299792.458

    # Calculate the systemic velocity corresponding to the redshift
    beta_sys = ((1 + z)**2 - 1) / ((1 + z)**2 + 1)
    v_sys = beta_sys * c

    # Use the relativistic velocity addition formula to get the velocity in the observer's frame
    v_obs = (vel + v_sys) / (1 + (vel * v_sys) / c**2)

    # Calculate the wavelength ratio from the observed velocity
    beta_obs = v_obs / c
    wavelength_ratio = np.sqrt((1 + beta_obs) / (1 - beta_obs))

    # Calculate the observed wavelength
    observed_wavelength = restLambda * wavelength_ratio

    return observed_wavelength

def get_data_dir():
    """
    Get the base data directory from the ASTRO_DATA_DIR environment variable.
    If not set, prompts the user to provide the path.
    Returns the path as a string.
    """
    
    data_dir = os.environ.get('ASTRO_DATA_DIR')
    if data_dir:
        return data_dir
    
    # Prompt user for the path
    print("\nASTRO_DATA_DIR environment variable not set.")
    print("Please provide the path to your astronomy data directory.")
    
    try:
        user_path = input("Enter data directory path: ").strip()
        if user_path:
            # Set the environment variable for this session
            os.environ['ASTRO_DATA_DIR'] = user_path
            print(f"Using: {user_path}")
            print(f"To make this permanent, add to your ~/.bashrc:")
            print(f"  export ASTRO_DATA_DIR={user_path}")
            return user_path
        else:
            raise ValueError("ASTRO_DATA_DIR is required but not provided.")
    except (EOFError, KeyboardInterrupt):
        raise ValueError("\nASTRO_DATA_DIR is required but not provided.")

def get_spectra_dir():
    """
    Get the R21 spectra directory from the R21_SPECTRA_DIR environment variable.
    If not set, prompts the user to provide the path.
    Returns the path as a string.
    """
    
    spectra_dir = os.environ.get('R21_SPECTRA_DIR')
    if spectra_dir:
        return spectra_dir
    
    # Prompt user for the path
    print("\nR21_SPECTRA_DIR environment variable not set.")
    print("Please provide the path to the R21 spectra directory.")
    print("(This should contain subdirectories for each cluster, e.g., A2744/, MACS0416/, etc.)")
    
    try:
        user_path = input("Enter R21 spectra directory path: ").strip()
        if user_path:
            # Set the environment variable for this session
            os.environ['R21_SPECTRA_DIR'] = user_path
            print(f"Using: {user_path}")
            print(f"To make this permanent, add to your ~/.bashrc:")
            print(f"  export R21_SPECTRA_DIR={user_path}")
            return user_path
        else:
            raise ValueError("R21_SPECTRA_DIR is required but not provided.")
    except (EOFError, KeyboardInterrupt):
        raise ValueError("\nR21_SPECTRA_DIR is required but not provided.")

def get_source_spectra_dir():
    """
    Get the source spectra directory from the SOURCE_SPECTRA_DIR environment variable.
    If not set, prompts the user to provide the path.
    Returns the path as a string.
    """
    
    source_dir = os.environ.get('SOURCE_SPECTRA_DIR')
    if source_dir:
        return source_dir
    
    # Prompt user for the path
    print("\nSOURCE_SPECTRA_DIR environment variable not set.")
    print("Please provide the path to the source spectra directory.")
    print("(This should contain subdirectories for each cluster with individual source spectra)")
    
    try:
        user_path = input("Enter source spectra directory path: ").strip()
        if user_path:
            # Set the environment variable for this session
            os.environ['SOURCE_SPECTRA_DIR'] = user_path
            print(f"Using: {user_path}")
            print(f"To make this permanent, add to your ~/.bashrc:")
            print(f"  export SOURCE_SPECTRA_DIR={user_path}")
            return user_path
        else:
            raise ValueError("SOURCE_SPECTRA_DIR is required but not provided.")
    except (EOFError, KeyboardInterrupt):
        raise ValueError("\nSOURCE_SPECTRA_DIR is required but not provided.")

def get_spectra_url():
    """
    Get the spectra base URL from the R21_URL environment variable.
    If not set, prompt the user to enter it interactively.
    Returns the URL as a string.
    Raises ValueError if not set and user does not provide one.
    """
    import os
    url = os.environ.get('R21_URL')
    if url:
        return url
    try:
        url = input("R21 spectra base URL not set. Please enter the URL: ").strip()
        if url:
            os.environ['R21_URL'] = url
            return url
        else:
            raise ValueError("R21 spectra base URL is required but not provided.")
    except Exception:
        raise ValueError("R21 spectra base URL is required but not provided.")
    
def load_spec(clus, iden, idfrom, spec_source = 'R21', spectype = 'weight_skysub'):
    """Loads a spectrum from the specified source
    clus: Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden: Identifier number of the object (e.g., 1234)
    idfrom: Prefix letter of the identifier (e.g., 'E' for E1234)
    spec_source: Source of the spectrum ('R21' for Richard et al. 2021, 'APER' for aperture spectra)
    spectype: Type of spectrum to load (either 'weight_syksub' or '2fwhm', etc.)
    """
    if spec_source == 'R21':
        return load_r21_spec(clus, iden, idfrom, spectype)
    elif spec_source == 'APER':
        return load_aper_spec(clus, iden, idfrom, spectype)
    else:
        raise ValueError(f"spec_source {spec_source} not recognized. Use 'R21' or 'APER'.")

def load_r21_spec(clus, iden, idfrom, spectype):
    """Loads a spectrum from the Richard et al. (2021) catalog
    clus: Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden: Identifier number of the object (e.g., 1234)
    idfrom: Prefix letter of the identifier (e.g., 'E' for E1234)
    spectype: Type of spectrum to load
    """
    # Generate the full identifier
    if iden[0].isdigit():
        identifier = (idfrom[0] + str(iden)).replace('E', 'X')
    elif iden[0].isalpha():
        identifier = iden
    else:
        raise ValueError("iden must start with a letter or digit")

    print(f"Loading R21 spectrum for {clus} object {identifier}...")

    # Get the spectra directory and construct the full path
    spectra_dir = get_spectra_dir()
    cluster_dir = os.path.join(spectra_dir, clus.upper())
    
    # Locate the file
    locfile = glob.glob(f"{cluster_dir}/spec_{identifier}_{spectype}.fits")
    if len(locfile) == 0: # If the file is not found, attempt to download it
        print(f"File not found. Downloading from {get_spectra_url()}{clus}_final_catalog/spectra/spec_{identifier}_{spectype}.fits")
        os.makedirs(cluster_dir, exist_ok=True)
        if clus == 'BULLET':
            os.system(f"wget --no-check-certificate {get_spectra_url()}Bullet_final_catalog/spectra/spec_{identifier}_{spectype}.fits"
                    + f" -P {cluster_dir}")
        else:
            os.system(f"wget --no-check-certificate {get_spectra_url()}{clus}_final_catalog/spectra/spec_{identifier}_{spectype}.fits"
                    + f" -P {cluster_dir}")
        print(f"Download complete.")
        locfile = glob.glob(f"{cluster_dir}/spec_{identifier}_{spectype}.fits")
    
    # If still not found, return None
    if len(locfile) == 0:
        print(f"File still not found after download attempt!")
        return None
    elif len(locfile) > 1:
        print(f"Multiple files found! Using the first one.")
    
    locfile = locfile[0] # Use the first match if multiple found

    # Load the spectrum as an astropy table
    def try_load(locfile):
        try:
            hdulist = fits.open(locfile)
            hdu = hdulist[1]
            varhdu = hdulist[2]
            spec = hdu.data
            errspec = np.sqrt(varhdu.data)
            wavelength = hdu.header['CRVAL1'] + np.arange(0., hdu.header['CDELT1'] * hdu.header['NAXIS1'], hdu.header['CDELT1'])
            return aptb.Table([wavelength, spec, errspec], names=('wave', 'spec', 'spec_err'))
        except Exception as e:
            print(f"Error loading FITS file: {e}")
            return None

    result = try_load(locfile)
    if result is None:
        print("File appears corrupted or truncated. Attempting to re-download...")
        # Remove the corrupted file
        try:
            os.remove(locfile)
        except Exception as e:
            print(f"Could not remove corrupted file: {e}")
        # Re-download
        if clus == 'BULLET':
            os.system(f"wget --no-check-certificate {get_spectra_url()}Bullet_final_catalog/spectra/spec_{identifier}_{spectype}.fits"
                    + f" -P {cluster_dir}")
        else:
            os.system(f"wget --no-check-certificate {get_spectra_url()}{clus}_final_catalog/spectra/spec_{identifier}_{spectype}.fits"
                    + f" -P {cluster_dir}")
        print("Re-download complete.")
        # Try loading again
        result = try_load(locfile)
        if result is None:
            print("File still corrupted after re-download!")
            return None
    return result

def load_aper_spec(clus, iden, idfrom, spectype = '2fwhm'):
    """Loads a spectrum from the aperture spectra files
    clus: Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden: Identifier number of the object (e.g., 1234)
    idfrom: Prefix letter of the identifier (e.g., 'E' for E1234)
    spectype: Type of spectrum to load ('2fwhm', '1fwhm', etc.)
    """
    # Generate the full identifier
    identifier = (idfrom[0] + str(iden)).replace('E', 'X')

    print(f"Loading aperture spectrum for {clus} object {identifier}...")

    # Get the source spectra directory and construct the full path
    source_dir = get_source_spectra_dir()
    cluster_dir = os.path.join(source_dir, clus.upper())
    
    # Locate the file
    locfile = glob.glob(f"{cluster_dir}/id{identifier}_{spectype}_spec.fits")
    if len(locfile) == 0:
        print(f"File not found!")
        return None
    elif len(locfile) > 1:
        print(f"Multiple files found! Using the first one.")
    
    locfile = locfile[0] # Use the first match if multiple found

    # Load the spectrum as an astropy table
    try:
        spectab = aptb.Table(fits.open(locfile)[1].data)
    except AttributeError:
        print("Error: The HDU does not have a .data attribute.")
        spectab = None

    return spectab


def is_reasonable_dpeak(popt, perr, z = None):
    """Check if the fitted parameters for a double peak Lyman-alpha profile are reasonable.
    Reasonable double peaked fits should be (1) statistically significant in both peaks,
    (2) be spectrally resolved, (3) have the blue peak at lower wavelength than the red peak,
    (4) have a velocity separation less than 1000 km/s.
    popt: List of fitted parameters from the lya_dpeak function
    perr: List of parameter uncertainties from the lya_dpeak function
    z: Redshift of the source (optional, used for velocity separation check)
    Returns True if the fit is reasonable, False otherwise.
    """
    # If z is not provided, get a rough estimate from the peak wavelength
    # of the red peak (popt[5])
    if z is None:
        z = popt[5] / 1215.67 - 1

    # Calculate 'SNR' values
    psnr = np.array(popt) / np.array(perr)
    
    # Calculate the right stddev of the blue and left stddev of the red lines to check 
    # that the lines are spectrally resolved
    fwhm_br = 2 * mdl.lya_swhm(ep.Complex(popt[2], perr[2]),
                                      ep.Complex(popt[3], perr[3]), +1)
    fwhm_rl = 2 * mdl.lya_swhm(ep.Complex(popt[6], perr[6]),
                                      ep.Complex(popt[7], perr[7]), -1)
    # Use the larger of the two widths as the minimum separation for the peaks
    wid = np.nanmax([np.abs(fwhm_br.value), np.abs(fwhm_rl.value)])
    # Catch cases where the blue peak has a poorly constrained width
    if fwhm_br.value < 3 * fwhm_br.error:
        wid = np.abs(fwhm_rl.value)
    
    # Calculate the velocity separation of the two peaks
    deltavel = np.abs(
        wave2vel(popt[1], 1215.67, redshift = z)
        - wave2vel(popt[5], 1215.67, redshift = z)
    )

    # Filter out bad results
    conditions = [
        (psnr[0] > 3.0 and psnr[4] > 3.0), # Both peaks are significant
        (psnr[0] < 5000 and psnr[4] < 5000), # Both peaks are not unphysically large
        popt[0] < popt[4],  # Blue peak amplitude < red peak amplitude
        np.abs(popt[5] - popt[1]) > wid, # Peaks are spectrally resolved
        popt[1] < popt[5], # Blue peak is at lower wavelength than red peak
        deltavel < 1000. # Velocity separation is less than 1000 km/s
    ]
    
    print(conditions)
    
    return all(conditions)

def get_line_spec(row, line, width, rest = True, spec_source = 'R21', spectype = 'weight_skysub'):
    """Extract a spectrum around a given line from a row of a catalog.
    row: Row of the catalog (astropy Table row)
    line: Name of the line (e.g., 'LYA', 'CIV', etc.)
    width: Width around the line to extract (in Angstroms)
    rest: If True, use rest-frame wavelength for the width; if False, use observed wavelength
    spec_source: Source of the spectrum ('R21' for Richard et al. 2021, 'APER' for aperture spectra)
    spectype: Type of spectrum to load (either 'weight_syksub' or '2fwhm', etc.)
    Returns wavelength array, flux array, and error array.
    """
    if line not in wavedict:
        raise ValueError(f"Line {line} not recognized. Available lines: {list(wavedict.keys())}")
    
    # Get the rest wavelength of the line
    rest_wavelength = wavedict[line]

    # See whether the line was detected in this object -- if so, use the line wavelength,
    # if not, use the LYMAN ALPHA redshift to get the expected wavelength
    snr_colname = f"SNR_{line}" if line != 'LYALPHA' else 'SNRR'
    lpeak_colname = f"LPEAK_{line}" if line != 'LYALPHA' else 'LPEAKR'
    if row[snr_colname] > 3.0 and not np.isnan(row[lpeak_colname]):
        line_wavelength = row[lpeak_colname]
    elif not np.isnan(row['LPEAKR']):
        line_wavelength = rest_wavelength * (row['LPEAKR'] / wavedict['LYALPHA'])
    else:
        print(f"No Lyman-alpha redshift available for object {row['iden']}. Cannot determine line wavelength.")
        return None, None, None
    
    # Load the spectrum
    spectab = None
    if spec_source == 'R21':
        spectab = load_r21_spec(row['CLUSTER'], row['iden'], row['idfrom'], spectype)
    elif spec_source.upper() == 'APER':
        spectab = load_aper_spec(row['CLUSTER'], row['iden'], row['idfrom'], spectype)

    if spectab is None:
        print(f"No spectrum found for {row['CLUSTER']} ID{row['iden']} in {spec_source} catalog.")
        return None, None, None
    
    # Determine the wavelength range to extract
    if rest:
        z = line_wavelength / rest_wavelength - 1
        obs_width = width * (1 + z)
    else:
        obs_width = width
    
    # Mask sky lines and bad data points
    mask = generate_spec_mask(spectab['wave'],
                                    spectab['spec'],
                                    spectab['spec_err'],
                                    lpeak_init = line_wavelength,
                                    continuum_buffer = obs_width,
                                    linename=line)

    return spectab['wave'][mask], spectab['spec'][mask], spectab['spec_err'][mask]


def find_partner(linename, linelist=None):
    """
    Find the doublet partner of a given spectral line.
    
    Parameters:
    -----------
    linename : str
        Name of the spectral line (e.g., 'CIV1548')
    linelist : list, optional
        List of line names to search for the partner. If None, uses all lines in doublets dict.
    
    Returns:
    --------
    str or None
        Name of the partner line, or None if no partner exists
    """
    if linelist is None:
        linelist = list(wavedict.keys())
    
    # Check each doublet
    for key, (line1, line2) in doubletdict.items():
        if linename == line1 and line2 in linelist:
            return line2
        elif linename == line2 and line1 in linelist:
            return line1
    
    return None


def flag_fitted_line(megatab, index, linename, spectab=None, 
                     fwhm_threshold=2.41, verbose=True):
    """
    Apply automatic quality flags to a fitted spectral line in a megatable.
    
    This function performs several tests on a fitted spectral line to identify
    potentially problematic measurements. Flags are applied as single characters:
    - 's': Sky line contamination
    - 't': Line too thin (FWHM below spectral resolution)
    - 'n': Negative flux domination (spectrum goes below zero in line region)
    - 'p': Peak-dominated (peak significance exceeds integrated flux SNR)
    
    The function modifies the table in place by updating the FLAG column for the line.
    
    Parameters:
    -----------
    megatab : astropy.table.Table
        Results table containing fitted line parameters. Must have columns:
        - LPEAK_{linename}: Peak wavelength of the line
        - LPEAK_ERR_{linename}: Error on peak wavelength
        - FLUX_{linename}: Integrated flux of the line
        - SNR_{linename}: Signal-to-noise ratio of the integrated flux
        - FWHM_{linename}: Full width at half maximum
        - FLAG_{linename}: Flag column (modified in place)
        Optional columns for improved error propagation:
        - FLUX_ERR_{linename}: Error on integrated flux (falls back to FLUX/SNR)
        - FWHM_ERR_{linename}: Error on FWHM (falls back to 0 if not provided)
    index : int
        Row index in the table to flag
    linename : str
        Name of the spectral line (e.g., 'CIV1548', 'LYALPHA')
    spectab : astropy.table.Table, optional
        Spectrum table with columns 'wave', 'spec', 'spec_err'. If None, peak
        significance test will be skipped. Required for complete flagging.
    fwhm_threshold : float, optional
        Minimum FWHM threshold in Angstroms (default: 2.41, MUSE spectral resolution)
    verbose : bool, optional
        If True, print diagnostic information about each test
    
    Returns:
    --------
    dict
        Dictionary of test results with keys:
        - 'sky': bool, True if sky line contamination detected
        - 'thin': bool, True if FWHM below threshold
        - 'negative': bool, True if spectrum dominated by negative flux
        - 'peakdominant': bool, True if peak dominates over integrated flux
        - 'contamination': bool, True if already flagged with 'c'
        - 'flags_applied': str, The flag string that was applied
    
    Notes:
    ------
    - If the line has a doublet partner that is also detected (SNR > 3), all flags
      are removed as the doublet provides confirmation of the detection.
    - Pre-existing 'c' (contamination) flags from other analyses are preserved.
    - Sky line contamination is checked against known sky line catalogs.
    - Peak amplitude uncertainty is computed using proper error propagation via the
      error_propagation package, accounting for uncertainties in both flux and FWHM.
      If error columns are not present, flux error is estimated from FLUX/SNR.
    
    Example:
    --------
    >>> from astropy.table import Table
    >>> import numpy as np
    >>> megatab = Table({'LPEAK_CIV1548': [5000.0],
    ...                  'LPEAK_ERR_CIV1548': [0.5],
    ...                  'FLUX_CIV1548': [1e-17],
    ...                  'SNR_CIV1548': [5.0],
    ...                  'FWHM_CIV1548': [2.0],
    ...                  'FLAG_CIV1548': ['']})
    >>> result = flag_fitted_line(megatab, 0, 'CIV1548')
    >>> print(megatab['FLAG_CIV1548'][0])
    't'
    """
    row = megatab[index]
    
    # Get column names for this line
    lpeak_col = f'LPEAK_{linename}'
    lpeak_err_col = f'LPEAK_ERR_{linename}'
    flux_col = f'FLUX_{linename}'
    snr_col = f'SNR_{linename}'
    fwhm_col = f'FWHM_{linename}'
    flag_col = f'FLAG_{linename}'
    
    # Initialize test results
    tests = {
        'sky': False,
        'thin': False,
        'peakdominant': False,
        'negative': False,
        'contamination': False,
        'flags_applied': ''
    }
    
    # Extract line parameters with errors where available
    lpeak = row[lpeak_col]
    lpeak_err = row[lpeak_err_col]
    flux = row[flux_col]
    snr = row[snr_col]
    fwhm = row[fwhm_col]
    current_flag = row[flag_col]
    
    # Get flux and FWHM errors if they exist
    flux_err_col = f'FLUX_ERR_{linename}' if f'FLUX_ERR_{linename}' in megatab.colnames else None
    fwhm_err_col = f'FWHM_ERR_{linename}' if f'FWHM_ERR_{linename}' in megatab.colnames else None
    
    flux_err = row[flux_err_col] if flux_err_col else np.abs(flux / snr)
    fwhm_err = row[fwhm_err_col] if fwhm_err_col else 0.0
    
    # Check for pre-existing contamination flag
    if current_flag == 'c':
        tests['contamination'] = True
    
    # Test 1: Sky line contamination
    # Check if the line peak coincides with any known sky lines
    tests['sky'] = check_sky_contamination(lpeak, lpeak_err, flux)
    
    # Test 2: Line too thin (below spectral resolution)
    tests['thin'] = (fwhm < fwhm_threshold)
    
    # Test 3: Negative flux domination
    # Check if the spectrum around the line is dominated by negative values
    if spectab is not None:
        # Define the region around the line (±2 FWHM or minimum 5 Å)
        half_width = np.max([2.0 * fwhm, 5.0])
        line_mask = (spectab['wave'] > lpeak - half_width) & (spectab['wave'] < lpeak + half_width)
        
        if np.sum(line_mask) > 0:
            line_flux = spectab['spec'][line_mask]
            # Check if more than 50% of pixels are negative
            frac_negative = np.sum(line_flux < 0) / len(line_flux)
            # Check if the median flux is negative
            median_negative = np.median(line_flux) < 0
            
            # Flag if either condition is met
            tests['negative'] = (frac_negative > 0.5) or median_negative
            
            if verbose and tests['negative']:
                print(f"Negative flux check: {frac_negative*100:.1f}% of pixels negative, median={np.median(line_flux):.2e}")
    
    # Test 4: Peak-dominated line
    # Calculate peak amplitude with proper error propagation
    if spectab is not None:
        # Use error_propagation to compute amplitude and its uncertainty
        # amp = flux / (sigma * sqrt(2*pi)), where sigma = fwhm / (2 * sqrt(2*ln(2)))
        flux_complex = ep.Complex(flux, flux_err)
        fwhm_complex = ep.Complex(fwhm, fwhm_err)
        
        # sigma = fwhm / 2.355
        sigma_complex = fwhm_complex / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        # amp = flux / (sigma * sqrt(2*pi))
        amp_complex = flux_complex / (sigma_complex * np.sqrt(2.0 * np.pi))
        
        # Get the continuum uncertainty at the line peak from the spectrum
        cont_err = np.interp(lpeak, spectab['wave'], spectab['spec_err'])
        
        # Peak significance is the ratio of amplitude to continuum uncertainty
        peak_snr = np.abs(amp_complex.value / cont_err)
        tests['peakdominant'] = (peak_snr > np.abs(snr))
        
        if verbose:
            print(f"Peak amplitude: {amp_complex.value:.2e} +/- {amp_complex.error:.2e}")
            print(f"Continuum error: {cont_err:.2e}")
            print(f"Peak SNR: {peak_snr:.2f}, Integrated SNR: {snr:.2f}")
    
    # Check for doublet partner
    partner = find_partner(linename, list(wavedict.keys()))
    if partner is not None:
        partner_snr_col = f'SNR_{partner}'
        partner_flag_col = f'FLAG_{partner}'
        # If both lines in doublet are detected, remove all flags
        if partner_snr_col in megatab.colnames:
            if np.abs(row[partner_snr_col]) > 3.0 and np.abs(snr) > 3.0:
                if verbose:
                    print(f"Doublet partner {partner} detected - removing flags")
                megatab[flag_col][index] = ''
                if partner_flag_col in megatab.colnames:
                    megatab[partner_flag_col][index] = ''
                tests['flags_applied'] = ''
                return tests
    
    # Apply flags (skip contamination as it's pre-existing)
    flag_string = ''
    if not tests['contamination']:
        # Clear any non-contamination flags
        megatab[flag_col][index] = ''
        
        # Apply new flags
        if tests['sky']:
            flag_string += 's'
            if verbose:
                print(f"  Applying flag 's': Sky line contamination detected")
        if tests['thin']:
            flag_string += 't'
            if verbose:
                print(f"  Applying flag 't': Line too thin (FWHM={fwhm:.2f} < {fwhm_threshold:.2f} Å)")
        if tests['negative']:
            flag_string += 'n'
            if verbose:
                print(f"  Applying flag 'n': Spectrum dominated by negative flux")
        if tests['peakdominant']:
            flag_string += 'p'
            if verbose:
                print(f"  Applying flag 'p': Peak-dominated line")
        
        megatab[flag_col][index] = flag_string
    
    tests['flags_applied'] = flag_string
    
    if verbose:
        print(f"\nFlagging results for {linename}:")
        if flag_string:
            print(f"  Flags applied: '{flag_string}'")
        else:
            print(f"  No flags raised")
    
    return tests


def check_sky_contamination(lpeak, lpeak_err, flux, sig=5):
    """
    Check if a spectral line coincides with known sky lines.
    
    Parameters:
    -----------
    lpeak : float
        Peak wavelength of the fitted line
    lpeak_err : float
        Error on the peak wavelength
    flux : float
        Integrated flux of the line
    sig : float, optional
        Significance threshold for sky line contamination (default: 5)
    
    Returns:
    --------
    bool
        True if sky line contamination is detected
    """
    # Check tolerance: use 3-sigma or minimum of 2.4 Angstroms
    tol = np.nanmax([lpeak_err * 3.0, 2.4])
    
    # Check strong sky lines first
    for skyline_wave in skylinedict.values():
        if (lpeak - tol < skyline_wave < lpeak + tol):
            return True
    
    # Note: Full sky line catalog checking would require loading an external catalog
    # For now, we only check against the strong lines defined in constants
    # Users can extend this by loading the full MUSE sky line catalog if needed
    
    return False
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

# Import constants
from .constants import c, wavedict, doublets
from . import models as mdl
from . import fitting as fit

lsf = LSF(typ='qsim_v1')

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
    Get the base data directory from environment variables.
    Checks in order: ASTRO_DATA_DIR, then defaults to ~/.astro_data
    Returns the path as a string.
    """
    import os
    from pathlib import Path
    
    data_dir = os.environ.get('ASTRO_DATA_DIR')
    if data_dir:
        return data_dir
    
    # Default to user's home directory
    default_dir = Path.home() / '.astro_data'
    return str(default_dir)

def get_spectra_dir():
    """
    Get the R21 spectra directory from environment variable or construct from base data dir.
    Returns the path as a string.
    """
    import os
    from pathlib import Path
    
    # Check for explicit override
    spectra_dir = os.environ.get('R21_SPECTRA_DIR')
    if spectra_dir:
        return spectra_dir
    
    # Otherwise construct from base data directory
    base_dir = get_data_dir()
    return str(Path(base_dir) / 'muse_catalogs' / 'spectra')

def get_source_spectra_dir():
    """
    Get the source spectra directory from environment variable or construct from base data dir.
    Returns the path as a string.
    """
    import os
    from pathlib import Path
    
    # Check for explicit override
    source_dir = os.environ.get('SOURCE_SPECTRA_DIR')
    if source_dir:
        return source_dir
    
    # Otherwise construct from base data directory
    base_dir = get_data_dir()
    return str(Path(base_dir) / 'source_spectra')

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
    mask = fit.generate_spec_mask(spectab['wave'],
                                    spectab['spec'],
                                    spectab['spec_err'],
                                    lpeak_init = line_wavelength,
                                    continuum_buffer = obs_width,
                                    linename=line)

    return spectab['wave'][mask], spectab['spec'][mask], spectab['spec_err'][mask]
"""
Input/Output utilities module.

This module provides functions to manage data directories, load spectra from different sources,
and handle environment variables for processing of MUSE spectra.
"""

import os
import glob
import numpy as np
from astropy.io import fits
import astropy.table as aptb

def get_data_dir():
    """
    Get the base data directory from the ASTRO_DATA_DIR environment variable.
    If not set, prompts the user to provide the path.

    Returns
    -------
    str
        The base data directory path.
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

    Returns
    -------
    str
        The R21 spectra directory path.
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

    Returns
    -------
    str
        The source spectra directory path.
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

    Returns
    -------
    str
        The spectra base URL.

    Raises
    ------
    ValueError
        If not set and user does not provide one.
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
    """
    Loads a spectrum from the specified source.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden : int
        Identifier number of the object (e.g., 1234)
    idfrom : str
        Prefix letter of the identifier (e.g., 'E' for E1234)
    spec_source : str
        Source of the spectrum ('R21' for Richard et al. 2021, 'APER' for aperture spectra)
    spectype : str
        Type of spectrum to load (either 'weight_skysub' or '2fwhm', etc.)

    Returns
    -------
    astropy.table.Table or None
        The loaded spectrum table, or None if loading failed.
    """
    if spec_source == 'R21':
        return load_r21_spec(clus, iden, idfrom, spectype)
    elif spec_source == 'APER':
        return load_aper_spec(clus, iden, idfrom, spectype)
    else:
        raise ValueError(f"spec_source {spec_source} not recognized. Use 'R21' or 'APER'.")

def load_r21_spec(clus, iden, idfrom, spectype):
    """
    Loads a spectrum from the Richard et al. (2021) catalog.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden : int
        Identifier number of the object (e.g., 1234)
    idfrom : str
        Prefix letter of the identifier (e.g., 'E' for E1234)
    spectype : str
        Type of spectrum to load

    Returns
    -------
    astropy.table.Table or None
        The loaded spectrum table, or None if loading failed.
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
    """
    Loads a spectrum from the aperture spectra files.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden : str
        Identifier string of the object (e.g., 'X1234')
    idfrom : str
        Prefix letter of the identifier (e.g., 'E' for E1234) - not used for aperture spectra
    spectype : str
        Type of spectrum to load ('2fwhm', '1fwhm', etc.)

    Returns
    -------
    astropy.table.Table or None
        The loaded spectrum table, or None if loading failed.
    """
    # For aperture spectra, use iden directly (already in correct format)
    identifier = str(iden)

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

def load_segmentation_map(clus, download_if_missing=False):
    """
    Load the R21 segmentation map for a given cluster.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    download_if_missing : bool, optional
        If True, attempt to download the segmentation map if not found locally. Default is False.

    Returns
    -------
    astropy.io.fits.HDUList or None
        The segmentation map HDUList, or None if loading failed.
    """
    data_dir = get_data_dir()
    segmap_path = os.path.join(data_dir, 'muse_data', f'{clus}', 'seg.fits')
    
    if not os.path.isfile(segmap_path):
        print(f"Segmentation map file not found: {segmap_path}")

        if not download_if_missing:
            return None

        # Attempt to download the segmentation map
        print(f"Attempting to download segmentation map for {clus}...")
        os.makedirs(os.path.dirname(segmap_path), exist_ok=True)
        try:
            os.system(f"wget --no-check-certificate {get_spectra_url()}{clus}_final_catalog/segmentation_maps/{clus}_segmentation.fits"
                      + f" -O {segmap_path}")
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading segmentation map: {e}")
            return None
    
    try:
        return fits.open(segmap_path)
    except Exception as e:
        print(f"Error loading segmentation map: {e}")
        return None
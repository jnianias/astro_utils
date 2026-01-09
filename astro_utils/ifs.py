"""
Module for extracting spectra from MUSE data cubes including aperture optimisation.
"""

from pathlib import Path
import numpy as np
from . import io
from . import image_processing as imp
from . import constants as const

wavedict = const.wavedict # Dictionary of important wavelengths

# Dictionary mapping detection method codes to descriptions
det_types = {
    'M': 'MUSELET',
    'X': 'EXTERNAL',
    'P': 'PRIOR'
}

def optimise_aperture(cube, ra, dec, full_iden, zlya, cluster, method='lya', kwargs={"nstruct":2, "niter":1}):
    """
    Optimises the aperture position for a source based on the specified method.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The MUSE data cube.
    ra : float
        Right Ascension of the source.
    dec : float
        Declination of the source.
    full_iden : str
        Full identifier of the source.
    zlya : float
        Redshift of the Lyman-alpha line.
    cluster : str
        Name of the cluster.
    method : str, optional
        Method for aperture optimisation. Default is 'lya'.
    kwargs : dict
        Additional keyword arguments for the peak_detection method: nstruct, niter

    Returns
    -------
    tuple
        Optimised (RA, DEC) coordinates.
    """
    # Determine detection method for this source
    dm = full_iden[0]

    lya_rest = wavedict['LYALPHA']  # Angstroms
    lya_obs = lya_rest * (1 + zlya)

    ra_opt, dec_opt = ra, dec

    if method == 'lya':
        print(f"Optimising aperture for source ID {full_iden} using Lyman-α peak detection.")
        ra_opt, dec_opt = find_lya_peak(cube, ra, dec, zlya, cluster, kwargs=kwargs)
    elif method == 'segmap':
        print(f"Optimising aperture for source ID {full_iden} using brightest continuum peak in segmentation map.")
        # Load segmentation map for the cluster
        seg_map = io.load_segmentation_map(cluster)
        ra_opt, dec_opt = imp.get_segmap_peak(seg_map, ra, dec)
    else:
        print(f"Warning: Unknown optimisation method '{method}'. Using original coordinates.")
    
    print(f"Optimised aperture position for source ID {full_iden}: RA={ra_opt}, DEC={dec_opt}")
    return ra_opt, dec_opt

def find_lya_peak(cube, ra, dec, zlya, kwargs={"nstruct":2, "niter":1}):
    """
    Finds the peak of Lyman-alpha emission in a narrowband image around the Lyman-alpha line.
    
    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The MUSE data cube.
    ra : float
        Right Ascension of the source.
    dec : float
        Declination of the source.
    zlya : float
        Redshift of the Lyman-alpha line.
    kwargs : dict
        Additional keyword arguments for the peak_detection method: nstruct, niter

    Returns
    -------
    tuple
        Optimised (RA, DEC) coordinates based on Lyman-alpha peak.
    """
    lya_rest = wavedict['LYALPHA']  # Angstroms
    lya_obs = lya_rest * (1 + zlya)

    # Extract narrowband image around Lyman-alpha line, using a 2.5 Angstrom bandwidth and 10 Angstrom
    # bands for the continuum, separated by 10 Angstrom margin from the line
    lya_nb_img = cube.get_image((lya_obs-1.25, lya_obs+1.25), subtract_off=True, margin=10, fband=4)

    # Limit to a region around the source to speed up peak detection
    lya_nb_img = lya_nb_img.subimage(center=(dec, ra), size=5.0)

    # Find the 3 sigma threshold for peak detection using the .var extension of the image
    threshold = 2 * np.sqrt(lya_nb_img.var.data)

    # Find peak positions in the narrowband image in pixel coordinates
    peak_locs = lya_nb_img.peak_detection(**kwargs, threshold=threshold)
    # If no peaks found, return original coordinates and raise a warning
    if len(peak_locs) == 0:
        print(f"Warning: No peaks found during aperture optimisation. "
              "Using original coordinates.")
        return ra, dec
    
    # Convert peak locations to world coordinates
    peak_locs_world = lya_nb_img.wcs.pix2sky(peak_locs)

    # Select the closest peak to the original RA, DEC position
    separations = np.sqrt((peak_locs_world[:,0] - ra)**2 + (peak_locs_world[:,1] - dec)**2)
    closest_peak_idx = np.argmin(separations)
    closest_peak = peak_locs_world[closest_peak_idx]
    ra_opt, dec_opt = closest_peak[1], closest_peak[0]
    return ra_opt, dec_opt

def extract_spectra(source_cat, aper_size, cluster_name, overwrite=False, optimise_apertures=False):
    """
    Extracts spectra for sources in the source catalog using aperture photometry and saves them to disk.

    Parameters
    ----------
    source_cat : list of dict
        List of source dictionaries containing source information.
    aper_size : int
        Aperture size in FWHM for spectrum extraction.
    cluster_name : str
        Name of the cluster (e.g., 'A2744').
    overwrite : bool, optional
        Whether to overwrite existing spectra. Default is False.
    optimise_apertures : bool, optional
        Whether to optimise apertures during extraction. Default is False.

    Returns
    -------
    dict
        Dictionary mapping source identifiers to their extracted spectrum file paths.
    """
    spec_paths = {}
    missing_sources = []

    for source in source_cat:
        iden = source['iden']
        idfrom = source['idfrom']
        spec_path = io.get_aper_spectra_dir(cluster_name)
        full_iden = f"{idfrom[0].replace('E', 'X')}{iden}"
        spec_name = f"{full_iden}_{aper_size}fwhm" + ("_opt" if optimise_apertures else "")
        spec_file = Path(spec_path) / f"{spec_name}.fits"
        if not spec_file.exists() or overwrite:
            missing_sources.append((source, spec_file, full_iden))
        else:
            print(f"Spectrum for source ID {full_iden} already exists; skipping extraction.")
            spec_paths[full_iden] = spec_file

    if not missing_sources:
        print("All spectra already exist; skipping extraction.")
        return spec_paths

    # Only load the cube if needed
    cube = io.load_muse_cube(cluster_name)
    for source, spec_file, full_iden in missing_sources:
        ra = source['RA']
        dec = source['DEC']
        # If optimise_apertures is True, center the aperture on the peak flux position within the
        # segmentation map of the source
        ra_opt, dec_opt = ra, dec
        if optimise_apertures:
            ra_opt, dec_opt = optimise_aperture(cube, ra, dec, full_iden, source['zlya'], cluster_name)
        print(f"Extracting spectrum for source ID {full_iden} at RA: {ra_opt}, DEC: {dec_opt}")
        # Extract the spectrum using MPDAF's built in aperture method for Cube objects
        spectrum = cube.aperture((dec_opt, ra_opt), aper_size / 2)
        io.save_spectrum(spectrum, spec_file)
        spec_paths[full_iden] = spec_file

    return spec_paths


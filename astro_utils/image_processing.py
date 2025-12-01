"""
Image processing utilities module.

This module provides functions for creating and processing astronomical images,
particularly from MUSE data cubes.
"""

import numpy as np
import glob
import os
from pathlib import Path
import astropy.units as u
from mpdaf.obj import Cube
from . import io
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.nddata import Cutout2D


def get_muse_cube_dir():
    """
    Get the MUSE data cube directory from environment variable or construct from base data dir.
    
    Returns
    -------
    Path
        Path to the directory containing MUSE data cubes.
    
    Notes
    -----
    Checks for MUSE_CUBE_DIR environment variable first. If not set, constructs path
    from ASTRO_DATA_DIR. Expected structure: $ASTRO_DATA_DIR/muse_data/
    """
    # Check for explicit override
    cube_dir = os.environ.get('MUSE_CUBE_DIR')
    if cube_dir:
        return Path(cube_dir)
    
    # Otherwise construct from base data directory
    from . import spectroscopy as spectro
    base_dir = spectro.get_data_dir()
    return Path(base_dir) / 'muse_data'


def make_muse_img(row, size, lcenter, width, cont=None, verbose=True):
    """
    Create a narrowband image from a MUSE data cube.
    
    Generates a narrowband image centered at a specific wavelength from MUSE cube data,
    with optional continuum subtraction from adjacent wavelength regions.
    
    Parameters
    ----------
    row : dict or astropy.table.Row
        Row containing target information with keys:
        - 'CLUSTER': cluster name
        - 'RA': right ascension in degrees
        - 'DEC': declination in degrees
    size : float
        Size of the image cutout in arcseconds. Will be adjusted if too close to cube edge.
    lcenter : float
        Central wavelength for the narrowband image in Angstroms.
    width : float
        Half-width of the wavelength window in Angstroms (image will span lcenter±width).
    cont : tuple of float, optional
        If provided, continuum will be subtracted. Should be a tuple (offset, width) where:
        - offset: wavelength offset from lcenter for continuum regions (in Angstroms)
        - width: width of the continuum regions (in Angstroms)
        Continuum is estimated from regions at lcenter±(offset±width).
        Default is None (no continuum subtraction).
    verbose : bool, optional
        If True, print progress messages. Default is True.
    
    Returns
    -------
    mpdaf.obj.Image
        Narrowband image (with continuum subtracted if cont is provided).
    
    Raises
    ------
    FileNotFoundError
        If no FITS cube files are found for the specified cluster.
    
    Notes
    -----
    The function looks for MUSE cube files in:
    $MUSE_CUBE_DIR/{cluster}/cube/*.fits
    
    or if MUSE_CUBE_DIR is not set:
    $ASTRO_DATA_DIR/muse_data/{cluster}/cube/*.fits
    
    The size is automatically adjusted to ensure it doesn't exceed the cube boundaries.
    
    Examples
    --------
    >>> # Create a simple narrowband image
    >>> img = make_muse_img(row, size=4.0, lcenter=5000.0, width=10.0)
    
    >>> # Create image with continuum subtraction
    >>> img = make_muse_img(row, size=4.0, lcenter=5000.0, width=10.0, 
    ...                     cont=(50.0, 10.0))
    """
    position = (row['DEC'], row['RA'])
    wl = lcenter
    clus = row['CLUSTER']

    if verbose:
        print(f"Loading {clus} cube...")

    # Find and open the cube
    cube_dir = get_muse_cube_dir()
    cube_files = glob.glob(str(cube_dir / clus / 'cube' / '*.fits'))
    if not cube_files:
        raise FileNotFoundError(f"No FITS cube files found for cluster {clus} in {cube_dir / clus / 'cube'}")
    
    musedata = Cube(cube_files[0])

    if verbose:
        print("Done.")
        print(f"Generating image...")

    # Get coordinate range and adjust size if needed to stay within cube boundaries
    co_range = musedata.get_range(unit_wave=u.AA, unit_wcs=u.deg)
    tightness = [
        np.nanmin([np.abs(position[0] - x) for x in [co_range[1], co_range[4]]]),
        np.nanmin([np.abs(position[1] - x) for x in [co_range[2], co_range[5]]])
    ]
    size = np.nanmin([size, np.nanmin(tightness) * 2 * 3600])

    # Create narrowband image
    img_line = musedata.get_image(wave=(wl - width, wl + width))
    img_line = img_line.subimage(position, size, unit_size=u.arcsec)

    # Optionally subtract continuum from adjacent regions
    if cont:
        try:
            img_cont_u = musedata.subcube(position, size, lbda=(wl + cont[0], wl + cont[1])).mean(axis=0)
            img_cont_l = musedata.subcube(position, size, lbda=(wl - cont[1], wl - cont[0])).mean(axis=0)
            if verbose:
                print("Continuum subtracted.")
            return img_line - 0.5 * (img_cont_u + img_cont_l)
        except Exception as e:
            if verbose:
                print(f"Warning: Continuum subtraction failed: {e}")
                print("Returning image without continuum subtraction.")
            return img_line
    else:
        return img_line

def show_segmentation_map(row, ax_in, return_array = False, size = 'auto', download_if_missing=False):
    """
    Overlay the R21 segmentation map on a given matplotlib axis.
    
    Parameters
    ----------
    row : dict or astropy.table.Row
        Row containing target information with keys:
        - 'CLUSTER': cluster name
        - 'RA': right ascension in degrees
        - 'DEC': declination in degrees
    ax_in : matplotlib.axes.Axes
        Matplotlib axis to overlay the segmentation map on.
    size : float or 'auto', optional
        Size of the cutout in arcseconds. If 'auto', uses the smallest box enclosing the entire segmentation map,
        down to a minimum of 2 arcseconds.
        Default is 'auto'.
    download_if_missing : bool, optional
        If True, attempts to download the segmentation map if not found locally.
        Default is False.
    
    Returns
    -------
    np.ndarray, optional
        If return_array is True, returns the segmentation map array for the cutout region.
        Otherwise, returns None.
    """

    position = (row['DEC'], row['RA']) # (dec, ra)
    clus = row['CLUSTER'] 
    id   = row['iden'] 
    idno = int(''.join(filter(str.isdigit, id))) # Extract numeric part of id

    # Load segmentation map HDUList with proper file handling
    with io.load_segmentation_map(clus, download_if_missing=download_if_missing) as seg_hdul:
        if seg_hdul is None:
            print(f"Segmentation map for cluster {clus} could not be loaded.")
            return None
        
        # Get the primary HDU
        segmap = seg_hdul[0].data
        segmap_header = seg_hdul[0].header
        segmap_wcs = WCS(segmap_header)

        # Determine size of cutout
        if size == 'auto':
            ys, xs = np.where(segmap == idno)
            if len(xs) == 0 or len(ys) == 0:
                print(f"Can't find segmentation map for object ID {id} in cluster {clus}.")
                return
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            segmap_cutout = segmap[y_min:y_max+1, x_min:x_max+1]
        else: # Use the Cutout2D utility to make the cutout
            cutout = Cutout2D(segmap, position, size*u.arcsec, wcs=segmap_wcs, mode='trim')
            segmap_cutout = cutout.data

        # Overlay segmentation map on provided axis
        ax_in.imshow(segmap_cutout, origin='lower', cmap='tab20', alpha=0.5)

        if return_array:
            return segmap_cutout
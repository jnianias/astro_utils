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
from astropy.coordinates import SkyCoord
from . import spectroscopy as spectro
from astropy.io import ascii
from astropy.convolution import convolve, Gaussian2DKernel

def get_muse_psf(clus):
    """
    Get the MUSE PSF FWHM for a given cluster from the PSF data file provided in the data directory.
    
    Parameters
    ----------
    clus : str
        Cluster name.
    
    Returns
    -------
    float
        PSF FWHM in arcseconds.
    """
    base_dir = io.get_data_dir()
    psf_file = Path(base_dir) / 'muse_fwhms.txt'
    # Read the psf data table using astropy ascii
    fwhmtb = np.loadtxt(psf_file, dtype={'names': ('CLUSTER', 'PSF_FWHM'), 'formats': ('U20', 'f4')}, skiprows=1)
    clusind = np.where(fwhmtb['CLUSTER'] == clus)[0]
    if len(clusind) == 0: # If the cluster is not found, raise an error
        raise ValueError(f"Cluster {clus} not found in PSF data file.")
    return fwhmtb['PSF_FWHM'][clusind[0]]


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
    cube_dir = io.get_muse_cube_dir()
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

def show_segmentation_mask(row, ax_in, return_cutout = False, size = 'auto', download_if_missing=False,
                           convolve_psf = None):
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
    convolve_psf : float or None, optional
        If provided, the segmentation map cutout will be convolved with a Gaussian PSF of this FWHM (in arcseconds).
        Default is None (no convolution).
    return_cutout : bool, optional
        If True, returns the Cutout2D object containing the segmentation map cutout.
        Default is False.
    
    Returns
    -------
    Cutout2D or None
        If return_array is True, returns the Cutout2D object containing the segmentation map cutout.
        Otherwise, returns None.
    """

    position = SkyCoord(ra=row['RA'], dec=row['DEC'], unit='deg')
    clus = row['CLUSTER'] 
    id   = row['iden'] 
    idno = int(''.join(filter(str.isdigit, id))) # Extract numeric part of id

    # Load segmentation map HDUList with proper file handling
    with io.load_segmentation_map(clus, download_if_missing=download_if_missing) as seg_hdul:
        if seg_hdul is None:
            print(f"Segmentation map for cluster {clus} could not be loaded.")
            raise FileNotFoundError(f"Segmentation map for cluster {clus} not found.")
        
        # Look for an extension called "DATA". If not present, revert to primary HDU
        if 'DATA' in seg_hdul:
            segmap = seg_hdul['DATA'].data
            segmap_header = seg_hdul['DATA'].header
        else:
            segmap = seg_hdul[0].data
            segmap_header = seg_hdul[0].header
        
        segmap_wcs = WCS(segmap_header)

        # Check to see whether the object ID exists in the segmentation map
        if idno not in segmap:
            # Check to see what the iden is at the source position
            pix_coords = segmap_wcs.world_to_pixel(position) # Convert world to pixel coordinates
            x_pix, y_pix = int(pix_coords[0]), int(pix_coords[1]) # Pixel coordinates
            id_at_position = -1
            if (0 <= x_pix < segmap.shape[1]) and (0 <= y_pix < segmap.shape[0]): # Check to make sure within array bounds
                id_at_position = segmap[y_pix, x_pix] # Value at that pixel
            else:
                raise ValueError(f"Source position {position.to_string('hmsdms')} is out of bounds for "
                                 f"segmentation map of cluster {clus}.")
            print(f"Object ID {id} not found in segmentation map for cluster {clus}. "
                  f"ID at source position is {id_at_position}. Using that instead.")
            idno = id_at_position
                
        # Determine size of cutout
        if size == 'auto':
            ys, xs = np.where(segmap == idno)
            if len(xs) == 0 or len(ys) == 0:
                print(f"Can't find segmentation region for object ID {id} in cluster {clus}.")
                return None
            # Determine the minimum enclosing box
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            # Convert pixel coordinates to world coordinates
            world_min = segmap_wcs.pixel_to_world(x_min, y_min)
            world_max = segmap_wcs.pixel_to_world(x_max, y_max)
            # Calculate size in arcseconds
            size_x = np.abs(world_max.ra.arcsec - world_min.ra.arcsec)
            size_y = np.abs(world_max.dec.arcsec - world_min.dec.arcsec)
            size = np.max([size_x, size_y, 2.0]) # Minimum size of 2 arcseconds
            
        # Use the Cutout2D utility to make the cutout
        cutout = Cutout2D(segmap, position, size*u.arcsec, wcs=segmap_wcs, mode='trim')
        cutout.data = cutout.data == idno

        # Optionally convolve with Gaussian PSF
        if convolve_psf is not None:
            pixel_scale = np.abs(cutout.wcs.pixel_scale_matrix[0,0]) * 3600.0 # arcsec/pixel
            sigma_pixels = (convolve_psf / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / pixel_scale
            kernel = Gaussian2DKernel(sigma_pixels)
            cutout.data = convolve(cutout.data.astype(float), kernel, fill_value=0.0)
            cutout.data = cutout.data > 0.1 # Binarize after convolution

        # Overlay contour outlining the segmentation map on the provided axis
        ax_in.contour(cutout.data, levels=[0.5], colors='red', linewidths=1.5, 
                      transform=ax_in.get_transform(WCS(cutout.wcs.to_header())))

        if return_cutout:
            return cutout # returns the cutout object containing vital WCS info
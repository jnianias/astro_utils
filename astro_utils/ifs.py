"""
Module for extracting spectra from MUSE data cubes including aperture optimisation.
"""

def optimise_aperture(cube, ra, dec, full_iden, zlya, cluster, method='auto'):
    """
    Optimises the aperture position by centering it on the peak flux position within the segmentation map.

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
        Method to use for optimisation ('auto', 'lyalpha', 'continuum'). Default is 'auto'.

    Returns
    -------
    tuple
        Optimised (RA, DEC) coordinates.
    """
    # Determine detection method for this source
    dm = full_iden[0]

    if dm in ['M', 'X']:
        print(f"{det_types[dm]}-type source; using narrowband Lyman-alpha image for aperture optimisation.")
        lya_rest = wavedict['LYALPHA']  # Angstroms
        lya_obs = lya_rest * (1 + zlya)
        # Extract narrowband image around Lyman-alpha line, using a 2.5 Angstrom bandwidth and 10 Angstrom
        # bands for the continuum, separated by 10 Angstrom margin from the line
        lya_nb_img = cube.get_image((lya_obs-1.25, lya_obs+1.25), subtract_off=True, margin=10, fband=4)
        # Find the peak position in the narrowband image
        peakdict = lya_nb_img.peak(center=(dec, ra), radius=3.0, dpix=3)
        ra_opt, dec_opt = peakdict['x'], peakdict['y']
        return ra_opt, dec_opt
    
    if dm == 'P':
        print(f"PRIOR-type source; using HST continuum image for aperture optimisation.")
        # Extract continuum image by averaging over the full wavelength range
        cont_img, mask = auip.get_weight_image(cluster)
        # Find the peak position in the continuum image
        peakdict = cont_img.peak(center=(dec, ra), radius=3.0, dpix=3)
        ra_opt, dec_opt = peakdict['x'], peakdict['y']

    print(f"Optimised aperture position for source ID {full_iden}: RA={ra_opt}, DEC={dec_opt}")
    return ra_opt, dec_opt

def 
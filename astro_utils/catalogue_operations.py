import astropy.table as aptb
import numpy as np
from astropy.io import fits
from astropy.table import Column, vstack
import astropy.units as u
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re
import glob
import os
from . import spectroscopy as spectro
from . import constants as const

class R21CatalogNotFoundError(Exception):
    """Exception raised when R21 catalog is not available for a cluster"""
    pass

def get_catalog_dir():
    """
    Get the catalog directory from environment variable or construct from base data dir.
    Returns the path as a Path object.
    """
    # Check for explicit override
    catalog_dir = os.environ.get('R21_CATALOG_DIR')
    if catalog_dir:
        return Path(catalog_dir)
    
    # Otherwise construct from base data directory
    base_dir = spectro.get_data_dir()
    return Path(base_dir) / 'muse_catalogs' / 'catalogs'

def load_r21_catalogue(cluster):
    """Load R21 lensing catalog for a given cluster from local cache or download if not present"""
    # Define the path to the local cache directory
    cache_dir = get_catalog_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Local file path
    local_path = cache_dir.glob(f"{cluster}_v?.?.fits")
    local_path = list(local_path)

    # If not found locally, attempt to download
    if len(local_path) == 0:
        print(f"R21 catalog for {cluster} not found locally. Attempting to download...")
        success = download_r21_catalogue(cluster)
        if not success:
            raise R21CatalogNotFoundError(f"R21 catalog for {cluster} could not be downloaded.")
        
    # Check again for the local file after download attempt
    local_path = cache_dir.glob(f"{cluster}_v?.?.fits")
    local_path = list(local_path)
    if len(local_path) == 0:
        raise R21CatalogNotFoundError(f"R21 catalog for {cluster} still not found after download attempt.")
    elif len(local_path) > 1:
        print(f"Multiple versions of R21 catalog found for {cluster}. Using the first one.")

    local_path = local_path[0]  # Use the first match if multiple found
    
    # Load and return the catalog as an astropy table
    try:
        r21_table = aptb.Table(fits.open(local_path)[1].data)
        return r21_table
    except Exception as e:
        raise R21CatalogNotFoundError(f"Failed to load R21 catalog for {cluster}: {e}")
        
def has_valid_mul(row):
    mul_value = row['MUL']
    return (mul_value is not None and 
            str(mul_value).strip() != '' and 
            str(mul_value).strip().lower() != 'none')

def get_pointing_suffix(cluster_name):
    """Extract pointing suffix from cluster name if it exists"""
    if cluster_name.endswith('S') or cluster_name.endswith('NE'):
        return cluster_name[-2:] if cluster_name.endswith('NE') else cluster_name[-1]
    return ''

def make_r21_catalogue_dict(clusters):
    """Create a dictionary of R21 lens tables for given clusters"""
    lenstables = {}
    for clus in clusters:
        try:
            lenstables[clus] = load_r21_catalogue(clus)
        except R21CatalogNotFoundError as e:
            print(e)
            lenstables[clus] = None
    return lenstables

def is_counterpart(row1, row2, lenstables, sigma=3.0, method='fit_match'):
    """Determine if two catalogue entries are counterparts based on specified method.
    row1, row2: Astropy table rows representing catalogue entries
    lenstables: Dictionary of R21 lens tables for clusters
    sigma: Number of sigma for matching criteria
    method: 'fit_match' or 'R21' to determine matching method
    Returns: (is_counterpart: bool, significance: float)
    """    
    # Immediately eliminate any cases where there's a big difference in z to speed things up
    if np.abs(row1['z'] - row2['z']) > 0.1:
        return False, np.inf
    
    if method == 'fit_match':
        match_params = ['LPEAKR', 'FWHMR', 'ASYMR']
        
        if row1['SNRB'] > 3.0 and row2['SNRB'] > 3.0:
            match_params.append('LPEAKB')
            match_params.append('FWHMB')
            match_params.append('ASYMB')
        
        matches = []
        for param in match_params:
            tv = np.abs(row1[param] - row2[param]) < sigma * np.sqrt(row1[param+'_ERR'] ** 2. + row2[param+'_ERR'] ** 2.)
            matches.append(tv)
        
        matches.append((row1['SNRR'] > 3.0) and (row2['SNRR'] > 3.0))
        
        if row1['SNRB'] > 3.0 and ~(row2['SNRB'] > 3.0):
            matches.append(row1['FLUXB'] * row2['FLUXR'] / row1['FLUXR'] < 3.0 * row2['FLUXR_ERR'])
        
        return all(matches), np.abs(row1['LPEAKR'] - row2['LPEAKR']) / np.sqrt(row1['LPEAKR_ERR'] ** 2. + row2['LPEAKR_ERR'] ** 2.)
    
    elif method.upper() == 'R21':
        iden1 = row1['iden']
        clus  = row1['CLUSTER'] if 'MACS0416' not in row1['CLUSTER'] else 'MACS0416'
        iden2 = row2['iden']
        
        try:
            # This will use the cached table or load it once
            r21_tab       = lenstables[clus]
            r21_tab_idens = np.array([x['idfrom'][0].replace('E','X') + str(x['iden']) for x in r21_tab])
            r21_tab_cluss = np.array([x['CLUSTER'] for x in r21_tab])
            r21_row1      = r21_tab[(r21_tab_idens == iden1) * (r21_tab_cluss == row1['CLUSTER'])][0]
            r21_row2      = r21_tab[(r21_tab_idens == iden2) * (r21_tab_cluss == row2['CLUSTER'])][0]
            
            # Find out if it belongs to a system; if not, just skip it
            if not has_valid_mul(r21_row1) or not has_valid_mul(r21_row2):
                return False, 0.0
            else:
                
                # If it does belong to a system, then figure out whether it's in the same system as row 2
                # Handle potential None or empty values safely
                mul1 = str(r21_row1['MUL']).strip()
                mul2 = str(r21_row2['MUL']).strip()

                # Split and clean the system names
                sysnames1 = [x.split('.')[0].strip() for x in mul1.split(',') if x.strip()]
                sysnames2 = [x.split('.')[0].strip() for x in mul2.split(',') if x.strip()]

                # Check if they share any system names
                common_systems = set(sysnames1) & set(sysnames2)
                return len(common_systems) > 0, \
                        np.abs(row1['LPEAKR'] - row2['LPEAKR']) \
                        / np.sqrt(row1['LPEAKR_ERR'] ** 2. + row2['LPEAKR_ERR'] ** 2.)
            
        except KeyError as e:
            # Fall back to fit matching if R21 not available
            print(f"No lens table available for {clus}, falling back to fit matching")
            return is_counterpart(row1, row2, lenstables, sigma=sigma, method='fit_match')
        


def download_r21_catalogue(cluster, dest_dir=None):
    """Download R21 lens catalog from server with case handling for Bullet cluster"""
    
    # Use provided destination directory or get from configuration
    if dest_dir is None:
        dest_dir = get_catalog_dir()
    else:
        dest_dir = Path(dest_dir)
    
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle the special case for Bullet cluster
    if cluster.upper() == 'BULLET':
        server_cluster_name = 'Bullet'  # Lowercase on server
    else:
        server_cluster_name = cluster   # Normal case on server

    # URL to scrape for available versions
    base_url = f"https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/catalogs/"
    print(f"Scraping directory listing: {base_url}")
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the latest versioned FITS file
    pattern = re.compile(rf"{cluster.lower()}_v\d+\.\d+\.fits", re.IGNORECASE)
    fits_file = None
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and pattern.fullmatch(href.lower()):
            fits_file = href
            print(f"Found catalogue file: {fits_file}")
            break

    if not fits_file:
        print("No matching FITS file found in directory listing.")
        return None

    # Download the file
    file_url = base_url + fits_file
    print(f"Downloading {file_url} ...")
    file_response = requests.get(file_url)
    dest_path = dest_dir / fits_file
    with open(dest_path, "wb") as f:
        f.write(file_response.content)
    print(f"Download complete: {dest_path}")

    return str(dest_path)


def get_muse_cand(iden, clus):
    """Get all MUSELET and PRIOR lines for a given iden in a given cluster
    Includes a provision for M2055 and M20355 from MACS0416S since these have had their IDs changed
    iden: Full identifier string (e.g., 'E1234', 'X5678')
    clus: Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    Returns: Astropy table of lines for the object, sorted by SNR
    """
    if iden in ['M2055', 'M20355'] and clus == 'MACS0416S':
        iden = iden[:-2] # Just remove the '55' at the end

    # Get the relevant cluster line table
    catalog_dir = get_catalog_dir()
    linetab = fits.open(glob.glob(str(catalog_dir / f'{clus}_v1.?_lines.fits'))[0])[1].data

    # Generate a column of full identifiers
    full_idens = np.array([x['idfrom'][0].replace('E','X') + str(x['iden']) for x in linetab])

    # Find the relevant rows
    objidx = (full_idens == iden)

    if np.sum(objidx) == 0:
        print(f"WARNING! No lines for {iden} in {clus}")
        return []
    else:
        # Generate a table of lines for this object, sorted by SNR
        rows = aptb.Table(linetab[objidx])
        rows.sort('SNR', reverse=True)
        lya_z = rows[rows['LINE'] == 'LYALPHA']['Z'].data[0]
        # Eliminate any lines that are not within 1000 km/s of the Lya redshift
        goodrows = np.abs(spectro.wave2vel(rows['LBDA_OBS'], rows['LBDA_REST'], redshift=lya_z) < 1000.)
        for i, row in enumerate(rows):
            if row['LINE'] in const.flines:
                rows['FAMILY'][i] = const.flines[const.flines == row['LINE']][0]
            elif row['LINE'] in const.slines:
                rows['FAMILY'][i] = const.flines[const.slines == row['LINE']][0]
        return rows[goodrows]
    

def get_line_table(iden, clus):
    """Get a table of emission lines from the R21 catalogue for a given source."""
    candidate_line_table = get_muse_cand(iden, clus)

    if len(candidate_line_table) == 0:
        # No lines found, return empty table
        print(f"No lines found for {iden} in {clus}")
        return aptb.Table()

    # Sort the table by SNR
    candidate_line_table.sort('SNR', reverse=True) # This must be an astropy table

    # Get rid of any repeats (these can be present due to different line families)
    line_table = aptb.unique(candidate_line_table, keys='LINE')

    # Get rid of Lyman alpha as we have already fit that
    line_table = line_table[line_table['LINE'] != 'LYALPHA']
    
    return line_table
    

def insert_fit_results(megatab, clus, iden, lya_results, other_results, avgmu, flags):
    """Insert fitting results into the megatab for a given source.
    megatab: an Astropy table, the megatab to insert results into
    clus: the cluster identifier
    iden: the source identifier
    lya_results: tuple of dictionaries (params, errors) from the Lya fitting (keys the same as colnames)
    other_results: dictionary of tuples of dictionaries (params, errors) from other line fittings,
                   keyed by line name. The keys of the param and error dictionaries need to be appended
                   with '_<line_name>' to match the megatab colnames.
    NB: This function modifies the megatab in place, and as such requires the use of row indices
    rather than row objects to insert the data.
    """
    # Find the index of the row to update
    row_index = np.where((megatab['CLUSTER'] == clus) & (megatab['iden'] == iden[1:]))[0]
    if len(row_index) == 0:
        print(f"Source {iden} in cluster {clus} not found in megatab.")
        return
    row_index = row_index[0]  # Get the single index value

    # Insert Lya results
    lya_params, lya_errors, _, reduced_chisq = lya_results
    for key in lya_params.keys():
        if key in megatab.colnames:
            megatab[key][row_index] = lya_params[key]
            megatab[key + '_ERR'][row_index] = lya_errors[key]

    # insert reduced chi-squared for Lya fit
    if 'RCHSQ' not in megatab.colnames:
        megatab['RCHSQ'] = np.full(len(megatab), np.nan)
    megatab['RCHSQ'][row_index] = reduced_chisq
    
    bluecols = ['AMPB', 'FLUXB', 'DISPB', 'FWHMB', 'ASYMB', 'LPEAKB']

    # insert SNR for Lya fit
    megatab['SNRR'][row_index] = lya_params['FLUXR'] / lya_errors['FLUXR']
    if 'SNRB' in lya_params and 'FLUXB' in lya_errors and not np.isnan(lya_params['FLUXB']) and not np.isnan(lya_errors['FLUXB']):
        megatab['SNRB'][row_index] = lya_params['FLUXB'] / lya_errors['FLUXB']
    else:
        megatab['SNRB'][row_index] = np.nan
        for colname in bluecols:
            megatab[colname][row_index]             = np.nan
            megatab[colname + '_ERR'][row_index]    = np.nan

    # Insert other line results
    for line_name, (params, errors, _, rchsq) in other_results.items():
        for key in params.keys():
            colname = f"{key}_{line_name}"
            errcolname = f"{key}_ERR_{line_name}"
            if colname in megatab.colnames:
                megatab[colname][row_index] = params[key]
                megatab[errcolname][row_index] = errors[key]
        megatab[f'RCHSQ_{line_name}'][row_index] = rchsq
        megatab[f'SNR_{line_name}'][row_index] = params['FLUX'] / errors['FLUX']
        # Re-insert any flags that were carried over from group members
        if line_name in flags:
            megatab[f'FLAG_{line_name}'][row_index] = 'c'
            
    # Update the magnification
    megatab['MU'][row_index] = avgmu

    # Finally, update the identifier with the 'S' prefix to indicate stacked spectrum
    megatab['iden'][row_index] = iden


def update_table(megatable, index, linename, params, param_errs, rchsq, flag=''):
    """
    Update a table with new fit parameters and statistics for a given index.
    
    For Lyman alpha fits, uses special column naming conventions (e.g., 'FLUXB', 'RCHSQ').
    For other lines, appends the line name to column names (e.g., 'FLUX_CIII', 'RCHSQ_CIII').
    SNR values are calculated automatically from flux and error parameters.
    
    Parameters
    ----------
    megatable : astropy.table.Table
        The table to update
    index : int
        The index of the row to update
    linename : str
        Name of the emission line (e.g., 'lya', 'CIII', 'OIII')
    params : dict
        Dictionary of fit parameters (e.g., {'FLUXB': 1.2, 'FLUXR': 3.4, 'CONT': 0.5})
    param_errs : dict
        Dictionary of fit parameter errors (e.g., {'FLUXB': 0.1, 'FLUXR': 0.2})
    rchsq : float
        Reduced chi-squared value of the fit
    flag : str, optional
        Quality flag for the fit (default: ''). Common flags:
        - 'm': Multiple comparable peaks detected
        - 's': Sky line contamination
        - 't': Line too thin
        - 'n': Negative flux domination
        - 'p': Peak-dominated
        - 'c': Contamination
        
    Notes
    -----
    For Lyman alpha (linename='lya' or 'Lya'), column names are:
        - Parameters: 'FLUXB', 'FLUXR', 'FWHMB', etc. (no suffix)
        - Errors: 'FLUXB_ERR', 'FLUXR_ERR', etc. (no line name suffix)
        - Stats: 'RCHSQ', 'SNRB', 'SNRR' (no line name suffix)
        - Flag: 'FLAG' (no line name suffix)
        - SNRB calculated from FLUXB / FLUXB_ERR
        - SNRR calculated from FLUXR / FLUXR_ERR
    
    For other lines, column names include the line name:
        - Parameters: 'FLUX_CIII', 'FWHM_CIII', etc.
        - Errors: 'FLUX_ERR_CIII', 'FWHM_ERR_CIII', etc.
        - Stats: 'RCHSQ_CIII', 'SNR_CIII', etc.
        - Flag: 'FLAG_CIII'
        - SNR calculated from FLUX / FLUX_ERR
    """
    # Check if this is a Lyman alpha fit
    is_lya = linename.lower() in ['lya', 'lyman_alpha', 'lyalpha']
    
    # Update parameters
    for key, value in params.items():
        if is_lya:
            # For Lyman alpha, use parameter name directly
            colname = key
        else:
            # For other lines, append line name
            colname = f"{key}_{linename}"
        
        if colname in megatable.colnames:
            megatable[colname][index] = value
    
    # Update parameter errors
    for key, value in param_errs.items():
        if is_lya:
            # For Lyman alpha, use parameter name with _ERR suffix
            colname = f"{key}_ERR"
        else:
            # For other lines, append ERR and line name
            colname = f"{key}_ERR_{linename}"
        
        if colname in megatable.colnames:
            megatable[colname][index] = value
    
    # Update reduced chi-squared
    if is_lya:
        rchsq_col = "RCHSQ"
    else:
        rchsq_col = f"RCHSQ_{linename}"
    
    if rchsq_col in megatable.colnames:
        megatable[rchsq_col][index] = rchsq
    
    # Calculate and update SNR values
    if is_lya:
        # For Lyman alpha, calculate separate blue and red SNR
        if 'FLUXB' in params and 'FLUXB' in param_errs:
            fluxb = params['FLUXB']
            fluxb_err = param_errs['FLUXB']
            snrb = fluxb / fluxb_err if fluxb_err > 0 else np.nan
            if "SNRB" in megatable.colnames:
                megatable["SNRB"][index] = snrb
        
        if 'FLUXR' in params and 'FLUXR' in param_errs:
            fluxr = params['FLUXR']
            fluxr_err = param_errs['FLUXR']
            snrr = fluxr / fluxr_err if fluxr_err > 0 else np.nan
            if "SNRR" in megatable.colnames:
                megatable["SNRR"][index] = snrr
    else:
        # For other lines, calculate SNR from flux and error if available
        if 'FLUX' in params and 'FLUX' in param_errs:
            flux = params['FLUX']
            flux_err = param_errs['FLUX']
            snr = flux / flux_err if flux_err > 0 else np.nan
            snr_col = f"SNR_{linename}"
            if snr_col in megatable.colnames:
                megatable[snr_col][index] = snr
    
    # Update flag column if provided
    if flag:
        if is_lya:
            flag_col = "FLAG"
        else:
            flag_col = f"FLAG_{linename}"
        
        if flag_col in megatable.colnames:
            megatable[flag_col][index] = flag


def is_true_emitter(tab, wavedict, sig=3.0, n=1, return_lines=False):
    """
    Identify true emission line sources in a spectroscopic catalog.
    
    This function checks for statistically significant emission line detections
    that are not flagged as problematic (e.g., contaminated by sky lines). A source
    is considered a "true emitter" if it has at least `n` emission lines detected
    above the specified significance threshold with valid flags.
    
    Parameters
    ----------
    tab : astropy.table.Table or astropy.table.Row
        Catalog table or single row containing spectroscopic measurements.
        Must contain columns for each line in wavedict with format:
        'SNR_{linename}' for signal-to-noise ratio and 'FLAG_{linename}' for flags.
    wavedict : dict
        Dictionary mapping line names (str) to rest-frame wavelengths (float).
        Lines named 'LYALPHA' are excluded from the check.
    sig : float, optional
        Significance threshold (in sigma) for line detection. Default is 3.0.
    n : int, optional
        Minimum number of significant emission lines required to classify
        as a true emitter. Default is 1.
    return_lines : bool, optional
        If True, returns a tuple of (boolean array, list of detected lines).
        If False, returns only the boolean array. Default is False.
    
    Returns
    -------
    bool or numpy.ndarray
        Boolean or array of booleans indicating whether each source is a true emitter
        (has >= n significant, unflagged emission line detections).
    list of lists, optional
        If return_lines=True, returns a list where each element is a list of
        line names detected for that source.
    
    Notes
    -----
    - Valid flags are empty strings ('') or 'na' (not available).
    - Lyman alpha ('LYALPHA') is explicitly excluded from the analysis.
    - For a single row input, returns a scalar boolean (or tuple with single-element list).

    """
    tv = np.zeros(len(tab['Z']) if isinstance(tab['Z'], (np.ndarray, aptb.Table.Column)) else 1).astype(int)
    trulist = [[] for r in tv]
    for line in wavedict:
        if line == 'LYALPHA':
            continue
        else:
            tv += ((tab[f'SNR_{line}'] > sig) *
                       ((tab[f'FLAG_{line}'] == '') + (tab[f'FLAG_{line}'] == 'na'))).astype(int)
            truidcs = np.where((tab[f'SNR_{line}'] > sig)
                        * ((tab[f'FLAG_{line}'] == '') + (tab[f'FLAG_{line}'] == 'na')))
            for idx in truidcs[0]:
                trulist[idx].append(line)
    return (tv >= n, trulist) if return_lines else tv >= n


def find_closest_cluster_member(row, maxdist=5.0 * u.arcsec, return_all=False):
    """
    Find the closest cluster member(s) to a target galaxy based on normalized distance.
    
    This function identifies surrounding galaxies most likely to have an impact on the
    fitting by looking at their properties and calculating a normalized distance metric
    (distance weighted by flux/magnitude).
    
    Parameters
    ----------
    row : dict or astropy.table.Row
        Row containing target galaxy information with keys:
        - 'CLUSTER': cluster name
        - 'RA': right ascension in degrees
        - 'DEC': declination in degrees
        - 'z': redshift
    maxdist : astropy.units.Quantity, optional
        Maximum angular distance to search for neighbors. Default is 5.0 arcsec.
    return_all : bool, optional
        If True, return all qualifying neighbors sorted by normalized distance.
        If False, return only the closest neighbor. Default is False.
    
    Returns
    -------
    astropy.table.Table or astropy.table.Row or list
        If return_all=True: Table of all qualifying neighbors sorted by NORMDIST,
                           or empty list if none found.
        If return_all=False: Row of the closest neighbor, or None if none found.
    
    Notes
    -----
    The normalized distance is calculated as: distance / sqrt(flux), where flux
    is derived from the F814W magnitude. This weights closer bright objects more
    heavily than distant faint objects.
    
    Filtering criteria:
    - Distance must be <= maxdist
    - Redshift must be <= 2.9
    - Redshift confidence (zconf) must be >= 2
    """
    clus = row['CLUSTER']
    ortab = load_r21_catalogue(clus)
    ortab.add_column(Column([np.inf for p in ortab], name='NORMDIST'))
    
    def normalised_distance(distance, magnitude):
        """Calculate normalized distance weighted by flux."""
        dist_arcsec = distance.to(u.arcsec)
        flux = 10 ** (-magnitude / 2.5)
        return dist_arcsec.value / np.sqrt(flux)

    target_ra = row['RA']
    target_dec = row['DEC']
    target_z = row['z']

    min_normdist = np.inf
    best_galaxy = None
    closelist = []

    for crow in ortab:
        galaxy_ra = crow['RA']
        galaxy_dec = crow['DEC']
        galaxy_z = crow['z']
        distance = np.sqrt((target_ra - galaxy_ra)**2 + (target_dec - galaxy_dec)**2) * u.deg
        normdist = normalised_distance(distance, crow['MAG_ISO_HST_F814W'])
        
        # Apply filtering criteria
        if distance.to(u.arcsec) > maxdist or galaxy_z > 2.9 or crow['zconf'] < 2:
            continue
        elif return_all:
            crow['NORMDIST'] = normdist
            closelist.append(crow)
        
        if normdist < min_normdist:
            min_normdist = normdist
            best_galaxy = crow

    if return_all and len(closelist) > 0:
        ctab = vstack(closelist)
        ctab.sort(keys='NORMDIST')
    elif return_all:
        ctab = []

    return ctab if return_all else best_galaxy

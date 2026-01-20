"""
Tests for the fitting utilities in astro_utils.fitting module.
To be run with pytest.
"""

import pytest
from astro_utils import fitting as aufit
from astro_utils import spectroscopy as auspec
from astro_utils import catalogue_operations as aucat

wavedict = auspec.wavedict

def test_get_initial_guesses_from_catalog():
    """
    Test the get_initial_guesses_from_catalog function to ensure it retrieves correct initial guesses from the catalog.
    """

    # Test case 1: Lyman alpha line from P594 in MACS0416NE
    full_iden = 'P594'
    cluster   = 'MACS0416NE'
    line_name = 'LYALPHA'

    line_table = aucat.get_muse_cand(full_iden, cluster)

    initial_guesses = aufit.get_initial_guesses_from_catalog(line_table, line_name, type='em')

    expected_guesses = {
        'AMPB'   :  380.14799202510267,
        'LPEAKB' :  5152.057737999559 - 7.5,
        'DISPB'  :  3.9649905781616326 / 2.355,
        'ASYMB'  :  -0.1,
        'AMPR'   :  380.14799202510267,
        'LPEAKR' :  5152.057737999559,
        'DISPR'  :  3.9649905781616326 / 2.355,
        'ASYMR'  :  0.1,
        'CONT'   :  6.827872333173079
    }

    assert initial_guesses == pytest.approx(expected_guesses, rel=1e-5)

# def test_gen_bounds():
#     """
#     Test the gen_bounds function to ensure it generates correct parameter bounds.
#     """

#     # Test case 1: automatic bounds for lLyman alpha
#     redshift = 4.0
#     initial_guesses = {
#         'FLUXB' :  
#         'LPEAKB': wavedict['LYALPHA'] * (1 + redshift) - 8,
#         '
#     }
#     expected_bounds = [(0.0, 10.0), (1.0, 5.0)]
#     bounds = aufit.gen_bounds(param_names, lower_limits, upper_limits)
#     assert bounds == expected_bounds

#     # Test case 2: curstom bounds
#     param_names = ['paramA', 'paramB']
#     lower_limits = [-5.0, -10.0]
#     upper_limits = [0.0, 0.0]
#     expected_bounds = [(-5.0, 0.0), (-10.0, 0.0)]
#     bounds = aufit.gen_bounds(param_names, lower_limits, upper_limits)
#     assert bounds == expected_bounds
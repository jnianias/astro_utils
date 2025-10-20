"""
Astro Utils - Astronomy utilities package
"""

# Import all modules to make them easily accessible
from . import catalogue_operations
from . import constants
from . import cosmology
from . import fitting
from . import image_processing
from . import io
from . import models
from . import plotting
from . import sky_tools
from . import spectroscopy

# Define what gets imported with "from astro_utils import *"
__all__ = [
    'catalogue_operations',
    'constants',
    'cosmology',
    'fitting',
    'image_processing',
    'io',
    'models',
    'plotting',
    'sky_tools',
    'spectroscopy',
]

__version__ = '0.1'

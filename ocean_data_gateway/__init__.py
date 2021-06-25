"""
Search through multiple ERDDAP and Axiom databases for datasets.
"""

import ast
import logging

from pathlib import Path

import cf_xarray as cfxr
import requests

from .gateway import Gateway
from .readers import axds, erddap, local


# from .utils import match_var


try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


base_path = Path.home() / ".ocean_data_gateway"
base_path.mkdir(exist_ok=True)

logs_path = base_path / "logs"
logs_path.mkdir(exist_ok=True)

catalogs_path = base_path / "catalogs"
catalogs_path.mkdir(exist_ok=True)

variables_path = base_path / "variables"
variables_path.mkdir(exist_ok=True)

files_path = base_path / "files"
files_path.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(message)s", "%a %b %d %H:%M:%S %Z %Y")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(formatter)

root_log_path = logs_path / f"{__name__}.log"
file_handler = logging.FileHandler(root_log_path)
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)

# logger.addHandler(stream_handler)
logger.addHandler(file_handler)
# warnings are captured and only output to file_handler
logging.captureWarnings(True)

# all available sources/readers
_SOURCES = [erddap, axds, local]

# important built-in options for readers
OPTIONS = {
    "erddap": {"known_server": ["ioos", "coastwatch"]},
    "axds": {"axds_type": ["platform2", "layer_group"]},
}

# Available keys for Gateway
keys_kwargs = [
    "approach",
    "parallel",
    "erddap",
    "axds",
    "local",
    "readers",
    "kw",
    "stations",
    "dataset_ids",
    "variables",
]


# For variable identification with cf-xarray
# custom_criteria to identify variables is saved here
# https://gist.github.com/kthyng/c3cc27de6b4449e1776ce79215d5e732
my_custom_criteria_gist = "https://gist.githubusercontent.com/kthyng/c3cc27de6b4449e1776ce79215d5e732/raw/1eeda7fe43976126235b710f4f733384fff2a788/my_custom_criteria.py"
response = requests.get(my_custom_criteria_gist)
my_custom_criteria = ast.literal_eval(response.text)
cfxr.accessor.set_options(my_custom_criteria)

# Principle variable list. These variable names need to match those in the gist.
# units
# QARTOD numbers for variables
var_def = {
    "temp": {
        "units": "degree_Celsius",
        "fail_span": [-100, 100],
        "suspect_span": [-10, 40],
    },
    "salt": {"units": "psu", "fail_span": [-10, 60], "suspect_span": [-1, 45]},
    "u": {"units": "m/s", "fail_span": [-10, 60], "suspect_span": [-1, 45]},
    "v": {"units": "m/s", "fail_span": [-10, 60], "suspect_span": [-1, 45]},
    "ssh": {"units": "m", "fail_span": [-10, 10], "suspect_span": [-2, 3]},
}

# REGEX = {
#     "temp": "temp|Celsius",
#     "salt": "salt|salinity|psu",
#     "vel": "u-momentum|u-velocity|v-momentum|v-velocity|velocity|u velocity|v velocity|m/s|meter second-1",
#     "ssh": "zeta|ssh|mld|elevation|sea_surface|sea surface",
#     "rho": "dense|density|kg/m^3",
#     "depths": "bathy|depths|bathymetry",
# }
#
# MED = {"air": "air|wind", "water": "water|sea"}

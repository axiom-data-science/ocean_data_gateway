"""
Search through multiple ERDDAP and Axiom databases for datasets.
"""

# these have to be imported here to prevent import order issues later
import cf_xarray as cfxr  # isort:skip
from cf_xarray.units import units  # isort:skip
import pint_xarray  # isort:skip

pint_xarray.unit_registry = units  # isort:skip
import ast  # noqa: E402
import logging  # noqa: E402

from pathlib import Path  # noqa: E402

import requests  # noqa: E402


from .utils import (  # isort:skip  # noqa: E402, F401
    Reader,
    load_data,
    resample_like,
    return_response,
)


from .gateway import Gateway  # isort:skip  # noqa: E402, F401
from .readers import axds, erddap, local  # isort:skip  # noqa: E402

from .vars import (  # isort:skip  # noqa: F401, E402
    all_variables,
    check_variables,
    search_variables,
    select_variables,
)

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
    # "dataset_ids",
    "variables",
    "criteria",
    "variables",
    "var_def",
]

# QARTOD defs
qcdefs = {"4": "FAIL", "1": "GOOD", "9": "MISSING", "3": "SUSPECT", "2": "UNKNOWN"}

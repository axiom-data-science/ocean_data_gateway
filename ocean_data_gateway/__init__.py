"""
Search through multiple ERDDAP and Axiom databases for datasets.
"""

import logging

from pathlib import Path

from .readers import axds, erddap, local


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

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
# warnings are captured and only output to file_handler
logging.captureWarnings(True)

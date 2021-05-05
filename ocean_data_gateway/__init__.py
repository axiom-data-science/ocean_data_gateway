try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


## make necessary directories ##

# make base dir in user home directory
import pathlib
path_base = pathlib.Path.home().joinpath('.ocean_data_gateway')
path_base.mkdir(parents=True, exist_ok=True)

# make subdirectories too
path_catalogs = path_base.joinpath('catalogs')
path_catalogs.mkdir(parents=True, exist_ok=True)
path_logs = path_base.joinpath('logs')
path_logs.mkdir(parents=True, exist_ok=True)
path_variables = path_base.joinpath('variables')
path_variables.mkdir(parents=True, exist_ok=True)


# import search.search
from .readers import erddap, axds, local
from .gateway import Gateway
# import ocean_data_gateway.readers.erddap
# import readers.axds
# import readers.local
#
# from .gateway import (gateway)
# import Data
# from .ErddapReader import (ErddapReader, region)
# from .axdsReader import (axdsReader)
# from .localReader import (localReader)

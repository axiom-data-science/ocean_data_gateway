"""
Make local test netcdf file
"""

import os

import numpy as np
import xarray as xr


def make_local_netcdf():
    fname = "tests/test_local.nc"
    if os.path.exists(fname):
        ds = xr.open_dataset(fname)
    else:
        ds = xr.Dataset()
        ds["time"] = ("time", np.arange(10), {"standard_name": "time"})
        ds["longitude"] = ("time", np.arange(10), {"standard_name": "longitude"})
        ds["latitude"] = ("time", np.arange(10), {"standard_name": "latitude"})
        ds["temperature"] = ("time", np.arange(10), {"units": "degree_Celsius"})
        ds.to_netcdf(fname)

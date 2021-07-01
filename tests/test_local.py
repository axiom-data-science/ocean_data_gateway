"""
Test reading local files.
"""

import os

import numpy as np
import xarray as xr

import ocean_data_gateway as odg


fname = "test_local.nc"
if os.path.exists(fname):
    ds = xr.open_dataset(fname)
else:
    ds = xr.Dataset()
    ds["time"] = ("dim", np.arange(10), {"standard_name": "time"})
    ds["longitude"] = ("dim", np.arange(10), {"standard_name": "longitude"})
    ds["latitude"] = ("dim", np.arange(10), {"standard_name": "latitude"})
    ds["temperature"] = ("dim", np.arange(10), {"units": "degree_Celsius"})
    ds.to_netcdf(fname)


def test_class_init():
    """can initialize local reader."""
    reader = odg.local.LocalReader()
    assert reader


# def test_region_init():
#     assert odg.local.regions()


def test_local_netcdf():
    """can local netcdf work."""

    filenames = [fname]
    data = odg.local.stations({"filenames": filenames[0]})

    assert data.dataset_ids == [fname]
    assert np.allclose(data.data[fname]["time"], np.arange(10))

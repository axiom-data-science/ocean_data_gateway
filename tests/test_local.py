"""
Test reading local files.
"""

import numpy as np
import xarray as xr

import ocean_data_gateway as odg


def test_class_init():
    """can initialize local reader."""
    reader = odg.local.LocalReader()
    assert reader


# def test_region_init():
#     assert odg.local.regions()


def test_local_netcdf():
    """can local netcdf work."""
    ds = xr.Dataset()
    ds["time"] = ("dim", np.arange(10), {"standard_name": "time"})
    ds["longitude"] = ("dim", np.arange(10), {"standard_name": "longitude"})
    ds["latitude"] = ("dim", np.arange(10), {"standard_name": "latitude"})
    ds["temperature"] = ("dim", np.arange(10), {"units": "degree_Celsius"})
    ds.to_netcdf("test_dataset.nc")

    filenames = ["test_dataset.nc"]
    data = odg.Gateway(
        approach="stations", readers=odg.local, local={"filenames": filenames[0]}
    )

    assert data.dataset_ids[0] == ["test_dataset.nc"]
    assert np.allclose(data.data[0]["test_dataset.nc"]["time"], np.arange(10))
    assert np.allclose(data.qc()[0]["test_dataset.nc"]["temperature_qc"], np.ones(10))

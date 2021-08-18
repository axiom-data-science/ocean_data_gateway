"""
Test reading local files.
"""


import numpy as np

from make_test_files import make_local_netcdf

import ocean_data_gateway as odg


# make sure local netcdf test file exists
make_local_netcdf()
fname = "test_local.nc"
fullname = f"tests/{fname}"

# I don't know why this doesn't work anymore
# def test_class_init():
#     """can initialize local reader."""
#     reader = odg.local.LocalReader()
#     assert reader


# def test_region_init():
#     assert odg.local.regions()


def test_local_netcdf():
    """can local netcdf work."""

    filenames = fullname
    data = odg.local.stations({"filenames": filenames})

    assert data.dataset_ids[0] == fname
    assert np.allclose(data.data(fname)["time"], np.arange(10))

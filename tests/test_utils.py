"""
Test utils functions.
"""

import pandas as pd
import xarray as xr

import ocean_data_gateway as odg


def test_resample_like():
    """Test utils resampling."""

    # Make two test Datasets
    ds_resample = xr.Dataset()
    dates1 = pd.date_range(start="2010-01-01", end="2011-01-01", periods=20)
    ds_resample["time"] = ("time", dates1, {"standard_name": "time"})

    ds_resample_to = xr.Dataset()
    dates2 = pd.date_range(start="2010-01-01", end="2011-01-01", periods=10)
    ds_resample_to["time"] = ("time", dates2, {"standard_name": "time"})

    # resample: downsample first ds to second ds
    ds_out = odg.utils.resample_like(ds_resample, ds_resample_to)

    assert (dates2 == ds_out.time.values).all()

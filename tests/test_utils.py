"""
Test utils functions.
"""
from unittest import mock

import pandas as pd
import pytest
import xarray as xr

import ocean_data_gateway as odg

from ocean_data_gateway import utils


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


@mock.patch("requests.get")
def test_fetch_criteria_expects_dict(requests_mock):
    url = "http://notavaliddomain.com"
    resp = mock.MagicMock()
    requests_mock.return_value = resp
    resp.json.return_value = {}
    utils.fetch_criteria(url)
    assert resp.raise_for_status.called

    resp.json.return_value = []
    with pytest.raises(ValueError):
        utils.fetch_criteria(url)

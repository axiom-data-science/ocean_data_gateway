#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
"""Unit tests for odg.Gateway."""
import numpy as np
import pandas as pd
import pint_xarray
import pytest
import xarray as xr
from make_test_files import make_local_netcdf

import ocean_data_gateway as odg
from ocean_data_gateway import readers
from ocean_data_gateway.gateway import Gateway
from ocean_data_gateway.readers import DataReader, ErddapReader
from ocean_data_gateway.readers.axds import stations as axds_stations
from ocean_data_gateway.readers.erddap import region as erddap_region
from ocean_data_gateway.readers.erddap import stations as erddap_stations
from ocean_data_gateway.readers.local import stations as local_stations

from cf_xarray.units import units  # isort:skip


pint_xarray.unit_registry = units  # isort:skip


# make sure local netcdf test file exists
make_local_netcdf()
fname = "test_local.nc"
fullname = f"tests/{fname}"

my_custom_criteria = {
    "temp": {"name": "(?i)temperature$"},
}

var_def = {
    "temp": {
        "units": "degree_Celsius",
        "fail_span": [-100, 100],
        "suspect_span": [-10, 40],
    },
}


def test_units():
    ds = xr.Dataset()
    ds["salt"] = ("dim", np.arange(10), {"units": "psu"})
    ds["lat"] = ("dim", np.arange(10), {"units": "degrees_north"})
    assert ds.pint.quantify()


def test_approach_default():
    search = odg.Gateway(kw={})
    assert search.kwargs_all["approach"] == "region"


def test_qc():
    """Test qc can return something with local test file."""

    filenames = fullname
    data = odg.Gateway(
        approach="stations",
        readers=odg.local,
        local={"filenames": filenames},
        criteria=my_custom_criteria,
        var_def=var_def,
    )
    data.dataset_ids
    assert isinstance(data.meta, pd.DataFrame)
    data[fname]
    assert (data.qc()[fname]["temperature_qc"] == np.ones(10)).all()


def test_qc_error():
    """Running QC without config or units should draw error."""

    filenames = fullname
    data = odg.Gateway(
        approach="stations", readers=odg.local, local={"filenames": filenames}
    )
    data.dataset_ids
    assert isinstance(data.meta, pd.DataFrame)
    data[fname]

    with pytest.raises(AssertionError):
        assert (data.qc()[fname]["temperature_qc"] == np.ones(10)).all()



def test_default_sources():
    gateway = odg.Gateway(
        approach="stations",
    )
    for source in gateway.sources:
        assert any(
            [isinstance(source, i)]
            for i in [axds_stations, erddap_stations, local_stations]
        )


def test_ioos_erddap_source():
    gateway = odg.Gateway(
        **{
            "approach": "stations",
            "erddap": {
                "known_server": "ioos",
                "variables": "salinity",
            },
            "readers": readers.erddap,
        }
    )
    assert len(gateway.sources) == 1
    assert all([isinstance(i, erddap_stations) for i in gateway.sources])
    assert gateway.sources[0].parallel is True
    assert gateway.sources[0].known_server == "ioos"
    assert gateway.sources[0].filetype == "netcdf"
    assert gateway.sources[0].kw == {"min_time": "1900-01-01", "max_time": "2100-12-31"}


def test_gateway_by_region():
    gateway = odg.Gateway(
        **{
            "kw": {
                "min_lon": -124.0,
                "max_lon": -123.0,
                "min_lat": 39.0,
                "max_lat": 40.0,
                "min_time": "2021-4-1",
                "max_time": "2021-4-2",
            },
            "approach": "region",
            "erddap": {
                "known_server": "ioos",
                "variables": "salinity",
            },
            "readers": readers.erddap,
        }
    )
    assert len(gateway.sources) == 1
    assert all([isinstance(i, erddap_region) for i in gateway.sources])
    assert gateway.sources[0].parallel is True
    assert gateway.sources[0].variables == ["salinity"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest
import xarray as xr  # noqa: E402

from make_test_files import make_local_netcdf

import ocean_data_gateway as odg


from cf_xarray.units import units  # isort:skip
import pint_xarray  # isort:skip

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

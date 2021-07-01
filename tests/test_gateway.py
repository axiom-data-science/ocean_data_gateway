from cf_xarray.units import units  # isort:skip
import pint_xarray  # isort:skip

pint_xarray.unit_registry = units  # isort:skip

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def test_units():
    ds = xr.Dataset()
    ds["salt"] = ("dim", np.arange(10), {"units": "psu"})
    ds["lat"] = ("dim", np.arange(10), {"units": "degrees_north"})
    assert ds.pint.quantify()

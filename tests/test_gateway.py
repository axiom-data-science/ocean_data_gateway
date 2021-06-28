import cf_xarray
from cf_xarray.units import units
import pint_xarray
pint_xarray.unit_registry = units

# import ocean_data_gateway as odg
import xarray as xr
import numpy as np


def test_units():
    ds = xr.Dataset()
    ds["salt"] = ("dim", np.arange(10), {"units": "psu"})
    ds["lat"] = ("dim", np.arange(10), {"units": "degrees_north"})
    assert ds.pint.quantify()

    
# def test_QC():
#     ds = xr.Dataset()
#     ds["salt"] = ("dim", np.arange(10), {"units": "psu"})
#     assert ds.pint.quantify()

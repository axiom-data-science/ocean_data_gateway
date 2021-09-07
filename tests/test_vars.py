import urllib.parse

import pandas as pd
import pytest

import ocean_data_gateway as odg

from make_test_files import make_local_netcdf


# make sure local netcdf test file exists
make_local_netcdf()
fname = "test_local.nc"
fullname = f"tests/{fname}"

my_custom_criteria = {
    "temp": {
        "name": "(?i)temperature$"
    },
}

var_def = {
    "temp": {
        "units": "degree_Celsius",
        "fail_span": [-100, 100],
        "suspect_span": [-10, 40],
    },
}


def test_criteria_gateway():
    """Test that nicknames and variable names get same results."""

    filenames = fullname
    data1 = odg.Gateway(
        approach="stations", readers=odg.local, local={"filenames": filenames},
        criteria=my_custom_criteria, var_def=var_def, variables='temp'
    )
    data2 = odg.Gateway(
        approach="stations", readers=odg.local, local={"filenames": filenames},
        variables='temp'
    )




def test_criteria_axds():
    pass


def test_criteria_erddap():
    pass


def test_variable_names_gateway():
    pass

@pytest.mark.slow
def test_all_variables_axds():
    df = odg.all_variables("axds")

    # test
    path_csv_fname = odg.variables_path.joinpath("axds_platform2_variable_list.csv")
    df_test = pd.read_csv(path_csv_fname, index_col="variable")

    assert df.equals(df_test)


@pytest.mark.slow
def test_all_variables_erddap():
    ioos = "http://erddap.sensors.ioos.us/erddap"
    df = odg.all_variables(ioos)

    # test
    server_name = urllib.parse.urlparse(ioos).netloc
    path_name_counts = odg.variables_path.joinpath(
        f"erddap_variable_list_{server_name}.csv"
    )
    df_test = pd.read_csv(path_name_counts, index_col="variable")

    assert df.equals(df_test)


@pytest.mark.slow
def test_search_variables():
    assert "Salinity" in odg.search_variables("axds", "sal").index


@pytest.mark.slow
def test_check_variables():
    server = "http://erddap.sensors.ioos.us/erddap"
    vars = ["salinity", "sea_water_practical_salinity"]
    assert odg.check_variables(server, vars) is None

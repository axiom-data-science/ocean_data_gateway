import urllib.parse

import pandas as pd
import pytest

import ocean_data_gateway as odg


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

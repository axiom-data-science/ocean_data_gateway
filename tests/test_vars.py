import urllib.parse

import pandas as pd
import pytest

import ocean_data_gateway as odg


my_custom_criteria = {
    "salt": {
        "standard_name": "sea_water_salinity$|sea_water_practical_salinity$",
        "name": "(?i)salinity(?!.*(soil|_qc))|(?i)sea_water_salinity$|(?i)sea_water_practical_salinity$|(?i)salinity$|(?i)salt$|(?i)sal$|(?i)s.sea_water_practical_salinity$",
    },
}

my_var_def = {
    "salt": {"units": "psu", "fail_span": [-10, 60], "suspect_span": [-1, 45]},
}

kw = {
    "min_time": "2019-1-1",
    "max_time": "2019-1-2",
    "min_lon": -99,
    "max_lon": -88,
    "min_lat": 20,
    "max_lat": 30,
}


@pytest.mark.slow
def test_criteria_local():
    """Test that can read in local criteria and var_def."""

    search = odg.Gateway(
        approach="region",
        kw=kw,
        criteria=my_custom_criteria,
        var_def=my_var_def,
        variables="salt",
    )
    assert "sea_water_practical_salinity" in search.sources[0].variables


@pytest.mark.slow
def test_criteria_nonlocal():
    """Test that can read in nonlocal criteria and var_def."""
    criteria = "https://gist.githubusercontent.com/kthyng/c3cc27de6b4449e1776ce79215d5e732/raw/af448937e4896535e36ef6522df8460e8f928cd6/my_custom_criteria.py"
    var_def = "https://gist.githubusercontent.com/kthyng/b8056748a811479460b6d5fc5cb5537b/raw/6b531cc5d3072ff6a4f5174f882d7d91d880cbf8/my_var_def.py"
    search = odg.Gateway(
        approach="region", kw=kw, criteria=criteria, var_def=var_def, variables="salt"
    )
    assert "sea_water_practical_salinity" in search.sources[0].variables


@pytest.mark.slow
def test_criteria_gateway():
    """Test that nicknames and variable names get same results."""

    data1 = odg.Gateway(
        approach="region",
        readers=odg.erddap,
        kw=kw,
        erddap={"known_server": "ioos"},
        criteria=my_custom_criteria,
        var_def=my_var_def,
        variables="salt",
    )
    data2 = odg.Gateway(
        approach="region",
        readers=odg.erddap,
        kw=kw,
        erddap={"known_server": "ioos"},
        variables="sea_water_practical_salinity",
    )

    assert data2.sources[0].variables[0] in data1.sources[0].variables


@pytest.mark.slow
def test_criteria_axds():
    """Can input criteria directly to reader"""

    kwargs = {
        "criteria": my_custom_criteria,
        "kw": kw,
        "approach": "region",
        "parallel": False,
        "variables": "salt",
        "axds_type": "platform2",
    }
    search = odg.axds.region(kwargs)
    assert search.variables


@pytest.mark.slow
def test_remove_variable():
    """Remove variable from list and update dataset_ids."""

    my_custom_criteria = {
        "salt": {
            "standard_name": "sea_water_salinity$|sea_water_practical_salinity$",
            "name": "(?i)salinity(?!.*(soil|_qc))|(?i)sea_water_salinity$|(?i)sea_water_practical_salinity$|(?i)salinity$|(?i)salt$|(?i)sal$|(?i)s.sea_water_practical_salinity$",
        },
        "ssh": {
            "standard_name": "sea_surface_height$|sea_surface_elevation|sea_surface_height_above_sea_level$",
            "name": "(?i)sea_surface_elevation(?!.*?_qc)|(?i)sea_surface_height_above_sea_level_geoid_mllw$|(?i)zeta$|(?i)Sea Surface Height(?!.*?_qc)|(?i)Water Surface above Datum(?!.*?_qc)",
        },
    }

    my_var_def = {
        "salt": {"units": "psu", "fail_span": [-10, 60], "suspect_span": [-1, 45]},
        "ssh": {"units": "m", "fail_span": [-10, 10], "suspect_span": [-3, 3]},
    }

    search = odg.Gateway(
        approach="region",
        kw=kw,
        criteria=my_custom_criteria,
        readers=odg.erddap,
        erddap=dict(known_server="ioos"),
        var_def=my_var_def,
        variables=["ssh", "salt"],
    )
    num_dataset_ids_start = len(search.dataset_ids)
    search.sources[0].variables.pop(0)
    search.sources[0].variables.pop(0)
    search.sources[0].variables.pop(0)
    num_dataset_ids_end = len(search.dataset_ids)
    assert num_dataset_ids_start > num_dataset_ids_end


def test_var_def_criteria():
    """Draw error for var_def having more variables that criteria."""

    # add another entry
    my_var_def["u"] = {"units": "m/s", "fail_span": [-10, 10], "suspect_span": [-5, 5]}

    with pytest.raises(AssertionError):
        assert odg.Gateway(
            approach="region", kw=kw, criteria=my_custom_criteria, var_def=my_var_def
        )


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

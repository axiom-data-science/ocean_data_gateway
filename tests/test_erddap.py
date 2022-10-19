import os
import shutil
import tempfile

from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_data_gateway as odg


# Makes type annotations clearer
Fixture = Generator


class MockErddap:
    """A mock ERDDAP client."""

    tmpdir = None

    def __init__(
        self, server: str, protocol: Optional[str] = None, response: str = "html"
    ):
        """Mock init."""
        pass

    def get_search_url(
        self, *args, search_for: str = "noaa_co_ops_8771013", **kwargs
    ) -> str:
        fd, name = tempfile.mkstemp(dir=self.tmpdir, suffix=".csv")
        templ = (Path(__file__).parent / "data/search.csv").read_text()
        data = templ.replace("DATASET_ID", search_for)
        with os.fdopen(fd, "r+") as f:
            f.write(data)
        return name

    def get_info_url(self, *args, **kwargs) -> str:
        return str(Path(__file__).parent / "data/info.csv")

    def get_download_url(self, *args, **kwargs) -> str:
        return "http://blahblah"

    def to_xarray(self, *args, **kwargs) -> pd.DataFrame:
        return load_testdata("co_ops_8771013.nc")


def load_testdata(key: str) -> xr.Dataset:
    pth = Path(__file__).parent / f"data/{key}"
    return xr.open_dataset(pth)


@pytest.fixture
def mock_erddap_client() -> Fixture[MockErddap, None, None]:
    pth_name = tempfile.mkdtemp()

    class CleanupMockErddap(MockErddap):
        tmpdir = pth_name

    yield CleanupMockErddap
    shutil.rmtree(pth_name)


def test_station_ioos_1dataset_id_alltime(mock_erddap_client):
    station = odg.erddap.stations(
        {
            "stations": "noaa_nos_co_ops_8771013",
            "erddap_client_class": mock_erddap_client,
        }
    )
    assert station.kw == {"min_time": "1900-01-01", "max_time": "2100-12-31"}
    assert station.dataset_ids == ["noaa_nos_co_ops_8771013"]


@pytest.mark.integration
def test_station_ioos_1dataset_id_specific():
    dataset_id = "noaa_nos_co_ops_8771013"
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    kwargs = {
        "stations": "noaa_nos_co_ops_8771013",
        "kw": kw,
        "parallel": False,
    }
    station = odg.erddap.stations(kwargs)
    assert station.kw == {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    assert isinstance(station.meta, pd.DataFrame)
    data = station.data()
    assert data[dataset_id] == station.data(dataset_id)
    assert isinstance(data[dataset_id], xr.Dataset)
    # assert isinstance(data["noaa_nos_co_ops_8771013"], pd.DataFrame)
    data_times = data[dataset_id].cf.isel({"T": [0, -1]}).cf["T"]
    known_times = [np.datetime64("2019-01-01"), np.datetime64("2019-01-02")]
    assert (data_times == known_times).all()


@pytest.mark.integration
def test_station_ioos_2dataset_ids():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    dataset_ids = ["noaa_nos_co_ops_8771013", "noaa_nos_co_ops_8774230"]
    stations = odg.erddap.stations({"stations": dataset_ids, "kw": kw})
    assert stations.dataset_ids == dataset_ids
    assert not stations.meta.empty


@pytest.mark.integration
def test_station_ioos_1station():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    stationname = "8771013"
    stations = odg.erddap.stations({"stations": stationname, "kw": kw})
    assert stations.dataset_ids == ["noaa_nos_co_ops_8771013"]
    assert not stations.meta.empty


@pytest.mark.integration
def test_station_ioos_2stations():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    dataset_ids = ["noaa_nos_co_ops_8771013", "noaa_nos_co_ops_8774230"]
    stationnames = ["8771013", "8774230"]
    stations = odg.erddap.stations({"stations": stationnames, "kw": kw})
    assert sorted(stations.dataset_ids) == dataset_ids
    assert not stations.meta.empty


# Slow on CI
@pytest.mark.slow
def test_region_coastwatch():
    kw = {
        "min_time": "2019-1-1",
        "max_time": "2019-1-2",
        "min_lon": -99,
        "max_lon": -88,
        "min_lat": 20,
        "max_lat": 30,
    }
    variables = ["water_u", "water_v"]
    region = odg.erddap.region(
        {"kw": kw, "variables": variables, "known_server": "coastwatch"}
    )
    assert "ucsdHfrE1" in region.dataset_ids
    # assert sorted(region.dataset_ids) == ['ucsdHfrE1', 'ucsdHfrE2', 'ucsdHfrE6']
    assert not region.meta.empty


@pytest.mark.slow
def test_station_coastwatch():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    dataset_id = "ucsdHfrE6"
    station = odg.erddap.stations(
        {
            "stations": dataset_id,
            "kw": kw,
            "parallel": False,
            "known_server": "coastwatch",
        }
    )
    assert station.kw == kw
    assert isinstance(station.meta, pd.DataFrame)


# too slow to use regularly with github actions
@pytest.mark.slow
def test_region_ioos():
    kw = {
        "min_time": "2019-1-1",
        "max_time": "2019-1-2",
        "min_lon": -95,
        "max_lon": -94,
        "min_lat": 27,
        "max_lat": 29,
    }
    # if the code can run with this, it can deal with having 2 variables input
    # but not both variables in the datasets
    variables = ["salinity", "sea_water_practical_salinity"]
    region = odg.erddap.region({"kw": kw, "variables": variables})
    assert "tabs_b" in region.dataset_ids
    assert not region.meta.empty

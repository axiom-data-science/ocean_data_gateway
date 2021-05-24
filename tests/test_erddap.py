import numpy as np
import pandas as pd

import ocean_data_gateway as odg


def test_station_ioos_1dataset_id_alltime():
    station = odg.erddap.stations({"dataset_ids": "noaa_nos_co_ops_8771013"})
    assert station.kw == {"min_time": "1900-01-01", "max_time": "2100-12-31"}
    assert station.dataset_ids == ["noaa_nos_co_ops_8771013"]


def test_station_ioos_1dataset_id():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    station = odg.erddap.stations({"dataset_ids": "noaa_nos_co_ops_8771013", "kw": kw})
    assert station.kw == {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    assert isinstance(station.meta, pd.DataFrame)
    data = station.data()
    assert isinstance(data["noaa_nos_co_ops_8771013"], pd.DataFrame)
    assert list(
        data["noaa_nos_co_ops_8771013"].index[[0, -1]].sort_values().values
    ) == [
        np.datetime64("2019-01-01T00:00:00.000000000"),
        np.datetime64("2019-01-02T00:00:00.000000000"),
    ]


def test_station_ioos_2dataset_ids():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    dataset_ids = ["noaa_nos_co_ops_8771013", "noaa_nos_co_ops_8774230"]
    stations = odg.erddap.stations({"dataset_ids": dataset_ids, "kw": kw})
    assert stations.dataset_ids == dataset_ids
    assert not stations.meta.empty


def test_station_ioos_1station():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    stationname = "8771013"
    stations = odg.erddap.stations({"stations": stationname, "kw": kw})
    assert stations.dataset_ids == ["noaa_nos_co_ops_8771013"]
    assert not stations.meta.empty


def test_station_ioos_2stations():
    kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
    dataset_ids = ["noaa_nos_co_ops_8771013", "noaa_nos_co_ops_8774230"]
    stationnames = ["8771013", "8774230"]
    stations = odg.erddap.stations({"stations": stationnames, "kw": kw})
    assert sorted(stations.dataset_ids) == dataset_ids
    assert not stations.meta.empty


# Slow on CI
# def test_region_coastwatch():
#     kw = {
#         "min_time": "2019-1-1",
#         "max_time": "2019-1-2",
#         "min_lon": -99,
#         "max_lon": -88,
#         "min_lat": 20,
#         "max_lat": 30,
#     }
#     variables = ["water_u", "water_v"]
#     region = odg.erddap.region(
#         {"kw": kw, "variables": variables, "known_server": "coastwatch"}
#     )
#     assert "ucsdHfrE1" in region.dataset_ids
#     # assert sorted(region.dataset_ids) == ['ucsdHfrE1', 'ucsdHfrE2', 'ucsdHfrE6']
#     assert not region.meta.empty


#
# def test_station_coastwatch():
#     kw = {"min_time": "2019-1-1", "max_time": "2019-1-2"}
#     dataset_id = "ucsdHfrE6"
#     station = odg.erddap.stations(
#         {
#             "dataset_ids": dataset_id,
#             "kw": kw,
#             "parallel": False,
#             "known_server": "coastwatch",
#         }
#     )
#     assert station.kw == kw
#     assert isinstance(station.meta, pd.DataFrame)
#
#
# def test_region_ioos():
#     kw = {
#         "min_time": "2019-1-1",
#         "max_time": "2019-1-2",
#         "min_lon": -95,
#         "max_lon": -94,
#         "min_lat": 27,
#         "max_lat": 29,
#     }
#     variables = ["sea_water_practical_salinity"]
#     region = odg.erddap.region({"kw": kw, "variables": variables})
#     assert "tabs_b" in region.dataset_ids
#     assert not region.meta.empty

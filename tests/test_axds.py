# from search.axdsReader import axdsReader
import numpy as np
import pandas as pd
import xarray as xr

import ocean_data_gateway as odg


# CHECK PARALLEL AND NOT
# CHECK CATALOG
# TEST inputting catalog file


# Test Platforms, stations
def test_station_platforms_1dataset_id_alltime():
    station = odg.axds.stations(
        {
            "axds_type": "platform2",
            "dataset_ids": "c61eecf1-1c0e-5287-b6fb-a92b51b14d54",
        }
    )
    assert station.kw == {"min_time": "1900-01-01", "max_time": "2100-12-31"}
    assert station.dataset_ids == ["c61eecf1-1c0e-5287-b6fb-a92b51b14d54"]


def test_station_platforms_1dataset_id():
    kw = {"min_time": "2020-8-1", "max_time": "2020-8-2"}
    dataset_ids = "c61eecf1-1c0e-5287-b6fb-a92b51b14d54"
    station = odg.axds.stations(
        {"axds_type": "platform2", "dataset_ids": dataset_ids, "kw": kw}
    )
    assert station.kw == {"min_time": "2020-8-1", "max_time": "2020-8-2"}
    assert station.dataset_ids == [dataset_ids]
    assert not station.meta.empty
    assert isinstance(station.meta, pd.DataFrame)
    data = station.data()
    assert isinstance(data[dataset_ids], pd.DataFrame)
    assert list(data[dataset_ids].index[[0, -1]].sort_values().values) == [
        np.datetime64("2020-08-01T00:00:00.000000000"),
        np.datetime64("2020-08-02T23:50:21.000000000"),
    ]


def test_station_platforms_2dataset_ids():
    kw = {"min_time": "2020-9-20", "max_time": "2020-9-20"}
    dataset_ids = [
        "c61eecf1-1c0e-5287-b6fb-a92b51b14d54",
        "7d4ea195-aeda-5c78-aa0d-d4f77ba0ad95",
    ]
    stations = odg.axds.stations(
        {"axds_type": "platform2", "dataset_ids": dataset_ids, "kw": kw}
    )
    assert stations.dataset_ids == dataset_ids


def test_station_platforms_1station():
    kw = {"min_time": "2020-8-1", "max_time": "2020-8-2"}
    stationname = "ng645-20200730T1909"
    dataset_id = "c61eecf1-1c0e-5287-b6fb-a92b51b14d54"
    stations = odg.axds.stations(
        {"axds_type": "platform2", "stations": stationname, "kw": kw}
    )
    assert stations.dataset_ids == [dataset_id]


# Test layer_groups, stations
def test_station_layer_group_1station_alltime():
    # 1 SFBOFS layer_group
    stations = ["04784baa-6be8-4aa7-b039-269f35e92e91"]
    # SFBOFS module uuid
    dataset_ids = ["03158b5d-f712-45f2-b05d-e4954372c1ce"]
    station = odg.axds.stations({"axds_type": "layer_group", "stations": stations[0]})
    assert station.kw == {"min_time": "1900-01-01", "max_time": "2100-12-31"}
    assert station._stations == stations
    assert station.dataset_ids == dataset_ids


# Slow on CI
# def test_station_layer_group_1station():
#     kw = {"min_time": "2021-4-1", "max_time": "2021-4-2"}
#     # 1 SFBOFS layer_group
#     stations = ["04784baa-6be8-4aa7-b039-269f35e92e91"]
#     # SFBOFS module uuid
#     dataset_ids = ["03158b5d-f712-45f2-b05d-e4954372c1ce"]
#     station = odg.axds.stations(
#         {"axds_type": "layer_group", "stations": stations, "kw": kw}
#     )
#     assert station.kw == kw
#     assert isinstance(station.meta, pd.DataFrame)
#     assert not station.meta.empty
# data = station.data()
# assert isinstance(data[dataset_ids[0]], xr.Dataset)
# assert list(data[dataset_ids[0]].ocean_time[[0, -1]].values) == [
#     np.datetime64("2021-04-01T00:00:00.000000000"),
#     np.datetime64("2021-04-02T23:00:00.000000000"),
# ]


# def test_station_layer_group_2dataset_ids():
#     kw = {"min_time": "2021-4-1", "max_time": "2021-4-2"}
#     # 2 SFBOFS layer_groups
#     stations = [
#         "04784baa-6be8-4aa7-b039-269f35e92e91",
#         "29bd4c08-db9e-45ba-94ef-8ec34868d855",
#     ]
#     # SFBOFS module uuid
#     dataset_ids = ["03158b5d-f712-45f2-b05d-e4954372c1ce"]
#     station = odg.axds.stations(
#         {"axds_type": "layer_group", "stations": stations, "kw": kw}
#     )
#     assert station._stations == stations
#     assert station.dataset_ids == dataset_ids
#
#
#
#
# # Test variables search and check
# def test_variables():
#     pass
#
#
# # Test Platforms, region
# def test_region_platforms_no_variables():
#     kw = {
#         "min_time": "2015-1-1",
#         "max_time": "2015-1-2",
#         "min_lon": -98,
#         "max_lon": -97.5,
#         "min_lat": 28.5,
#         "max_lat": 29,
#     }
#     dataset_ids = [
#         "1b046f4e-5dd6-5ba7-8d70-743e2edeb1df",
#         "7d196e00-4d79-5d4a-a978-f56161d758ce",
#     ]
#     region = odg.axds.region({"kw": kw})
#     assert sorted(region.dataset_ids) == dataset_ids
#     assert region.kw == kw
#
#
# def test_region_platforms_variables():
#     kw = {
#         "min_time": "2019-1-1",
#         "max_time": "2021-1-2",
#         "min_lon": -98,
#         "max_lon": -95,
#         "min_lat": 28,
#         "max_lat": 29,
#     }
#     dataset_ids = ["612c17e4-3306-5d51-abb2-7e890fe49896"]
#     variables = ["Salinity"]
#     region = odg.axds.region({"kw": kw, "variables": variables})
#     assert dataset_ids[0] in region.dataset_ids
#     assert region.kw == kw
#     assert region.variables == variables


# Test layer_groups, region

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities to help with running the other code.
"""

import pandas as pd
import requests


def resample_like(ds_resample, ds_resample_to):
    """Resample input dataset in time like other dataset.

    Parameters
    ----------
    ds_resample: xarray Dataset
        Dataset to resample
    ds_resample_to: xarray Dataset
        Resample ds_resample to be at same times as ds_resample_to

    Returns
    -------
    ds_resample with times matches ds_resample_to.

    TODO: Does this work for upsampling as well as downsampling?
    TODO: Should the index be taken from ds_resample_to and example matched by
    reindexing?
    """

    # Find delta T for model (lower temporal resolution)
    dt = pd.Timestamp(ds_resample_to.cf["T"][1].values) - pd.Timestamp(
        ds_resample_to.cf["T"][0].values
    )
    dt = dt.total_seconds() / 3600  # delta t in hours

    # resample higher res data set to dt
    ds_resampled = ds_resample.resample(time=f"{dt}H").interpolate("linear")

    return ds_resampled


def fetch_criteria(url: str) -> dict:
    """Return parsed criteria dictionary from URL."""
    # For variable identification with cf-xarray
    # custom_criteria to identify variables is saved here
    # https://gist.github.com/kthyng/c3cc27de6b4449e1776ce79215d5e732
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

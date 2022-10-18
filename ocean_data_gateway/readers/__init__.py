#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Readers available for ocean_data_gateway.
"""
from .reader import Reader
from .data_reader import DataReader
from .erddap import ErddapReader
from .axds import AxdsReader
from .local import LocalReader

__all__ = ["Reader", "DataReader", "ErddapReader", "AxdsReader", "LocalReader"]

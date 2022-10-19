#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Readers available for ocean_data_gateway.
"""
from .axds import AxdsReader
from .data_reader import DataReader
from .erddap import ErddapReader
from .local import LocalReader
from .reader import Reader


__all__ = ["Reader", "DataReader", "ErddapReader", "AxdsReader", "LocalReader"]

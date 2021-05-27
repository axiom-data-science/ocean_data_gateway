"""
Reader for ERDDAP servers.
"""

import logging
import multiprocessing
import re

import numpy as np
import pandas as pd
import xarray as xr

from erddapy import ERDDAP
from joblib import Parallel, delayed

import ocean_data_gateway as odg


logger = logging.getLogger(__name__)

# this can be queried with
# search.ErddapReader.reader
reader = "erddap"


class ErddapReader:
    """
    This class searches ERDDAP servers. There are 2 known_servers but
    others can be input too.

    Attributes
    ----------
    parallel: boolean
        If True, run with simple parallelization using `multiprocessing`.
        If False, run serially.
    known_server: string
        Two ERDDAP servers are built in to be known to this reader: "ioos" and
        "coastwatch".
    e: ERDDAP server instance
    e.protocol: string
        * "tabledap" (pandas, appropriate for reading as csv)
        * "griddap" (xarray, appropriate for reading as netcdf)
    e.server: string
        Return the server name
    columns: list
        Metadata columns
    name: string
        "erddap_ioos", "erddap_coastwatch", or a constructed string if the user
        inputs a new protocol and server.
    reader: string
        reader is defined as "ErddapReader".
    """

    def __init__(self, known_server="ioos", protocol=None, server=None, parallel=True):
        """
        Parameters
        ----------
        known_server: string, optional
            Two ERDDAP servers are built in to be known to this reader:
            "ioos" and "coastwatch".
        protocol, server: string, optional
            For a user-defined ERDDAP server, input the protocol as one of the
            following:
            * "tabledap" (pandas, appropriate for reading as csv)
            * "griddap" (xarray, appropriate for reading as netcdf)
            and the server address (such as
            "http://erddap.sensors.ioos.us/erddap" or
            "http://coastwatch.pfeg.noaa.gov/erddap").
        parallel: boolean
            If True, run with simple parallelization using `multiprocessing`.
            If False, run serially.
        """
        self.parallel = parallel

        # either select a known server or input protocol and server string
        if known_server == "ioos":
            protocol = "tabledap"
            server = "http://erddap.sensors.ioos.us/erddap"
        elif known_server == "coastwatch":
            protocol = "griddap"
            server = "http://coastwatch.pfeg.noaa.gov/erddap"
        elif known_server is not None:
            statement = (
                "either select a known server or input protocol and server string"
            )
            assert (protocol is not None) & (server is not None), statement
        else:
            known_server = server.strip("/erddap").strip("http://").replace(".", "_")
            statement = (
                "either select a known server or input protocol and server string"
            )
            assert (protocol is not None) & (server is not None), statement

        self.known_server = known_server
        self.e = ERDDAP(server=server)
        self.e.protocol = protocol
        self.e.server = server

        # columns for metadata
        self.columns = [
            "geospatial_lat_min",
            "geospatial_lat_max",
            "geospatial_lon_min",
            "geospatial_lon_max",
            "time_coverage_start",
            "time_coverage_end",
            "defaultDataQuery",
            "subsetVariables",  # first works for timeseries sensors, 2nd for gliders
            "keywords",  # for hf radar
            "id",
            "infoUrl",
            "institution",
            "featureType",
            "source",
            "sourceUrl",
        ]

        # name
        self.name = f"erddap_{known_server}"

        self.reader = "ErddapReader"

    def find_dataset_id_from_station(self, station):
        """Find dataset_id from station name.

        Parameters
        ----------
        station: string
            Station name for which to search for dataset_id
        """

        if station is None:
            return None
        # for station in self._stations:
        # if station has more than one word, AND will be put between
        # to search for multiple terms together.
        url = self.e.get_search_url(
            response="csv", items_per_page=5, search_for=station
        )

        try:
            df = pd.read_csv(url)
        except Exception as e:
            logger.exception(e)
            logger.warning(f"search url {url} did not work for station {station}.")
            return

        # first try for exact station match
        try:
            # Special case for TABS when don't split the id name
            if "tabs" in station:  # don't split
                dataset_id = [
                    dataset_id
                    for dataset_id in df["Dataset ID"]
                    if station.lower() == dataset_id.lower()
                ][0]
            else:
                dataset_id = [
                    dataset_id
                    for dataset_id in df["Dataset ID"]
                    if station.lower() in dataset_id.lower().split("_")
                ][0]

        except Exception as e:
            logger.exception(e)
            logger.warning(
                "When searching for a dataset id to match station name %s, the first attempt to match the id did not work."
                % (station)
            )
            # If that doesn't work, return None for dataset_id
            dataset_id = None
            # # if that doesn't work, trying for more general match and just take first returned option
            # dataset_id = df.iloc[0]["Dataset ID"]

        return dataset_id

    @property
    def dataset_ids(self):
        """Find dataset_ids for server.

        Notes
        -----
        The dataset_ids are found by querying the metadata through the ERDDAP server. Or, if running with `stations()` and input dataset_ids, they are simply set initially with those values.
        """

        if not hasattr(self, "_dataset_ids"):

            # This should be a region search
            if self.approach == "region":

                # find all the dataset ids which we will use to get the data
                # This limits the search to our keyword arguments in kw which should
                # have min/max lon/lat/time values
                dataset_ids = []
                if self.variables is not None:
                    for variable in self.variables:

                        # find and save all dataset_ids associated with variable
                        search_url = self.e.get_search_url(
                            response="csv",
                            **self.kw,
                            variableName=variable,
                            items_per_page=10000,
                        )

                        try:
                            search = pd.read_csv(search_url)
                            dataset_ids.extend(search["Dataset ID"])
                        except Exception as e:
                            logger.exception(e)
                            logger.warning(
                                f"variable {variable} was not found in the search"
                            )
                            logger.warning(f"search_url: {search_url}")

                else:

                    # find and save all dataset_ids associated with variable
                    search_url = self.e.get_search_url(
                        response="csv", **self.kw, items_per_page=10000
                    )

                    try:
                        search = pd.read_csv(search_url)
                        dataset_ids.extend(search["Dataset ID"])
                    except Exception as e:
                        logger.exception(e)
                        logger.warning("nothing found in the search")
                        logger.warning(f"search_url: {search_url}")

                # only need a dataset id once since we will check them each for all standard_names
                self._dataset_ids = list(set(dataset_ids))

            # This should be a search for the station names
            elif self.approach == "stations":

                # search by station name for each of stations
                if self.parallel:
                    # get metadata for datasets
                    # run in parallel to save time
                    num_cores = multiprocessing.cpu_count()
                    dataset_ids = Parallel(n_jobs=num_cores)(
                        delayed(self.find_dataset_id_from_station)(station)
                        for station in self._stations
                    )

                else:
                    dataset_ids = []
                    for station in self._stations:
                        dataset_ids.append(self.find_dataset_id_from_station(station))

                # In this case return all dataset_ids so they match 1-1 with
                # the input station list.
                self._dataset_ids = dataset_ids

            else:
                logger.warning(
                    "Neither stations nor region approach were used in function dataset_ids."
                )

        return self._dataset_ids

    def meta_by_dataset(self, dataset_id):
        """Return the catalog metadata for a single dataset_id."""

        info_url = self.e.get_info_url(response="csv", dataset_id=dataset_id)
        try:
            info = pd.read_csv(info_url)
        except Exception as e:
            logger.exception(e)
            logger.warning(f"Could not read info from {info_url}")
            return {dataset_id: []}

        items = []

        for col in self.columns:

            try:
                item = info[info["Attribute Name"] == col]["Value"].values[0]
                dtype = info[info["Attribute Name"] == col]["Data Type"].values[0]
            except:
                if col == "featureType":
                    # this column is not present in HF Radar metadata but want it to
                    # map to data_type, so input 'grid' in that case.
                    item = "grid"
                else:
                    item = "NA"

            if dtype == "String":
                pass
            elif dtype == "double":
                item = float(item)
            elif dtype == "int":
                item = int(item)
            items.append(item)

        ## include download link ##
        self.e.dataset_id = dataset_id
        if self.e.protocol == "tabledap":
            if self.variables is not None:
                self.e.variables = [
                    "time",
                    "longitude",
                    "latitude",
                    "station",
                ] + self.variables
            # set the same time restraints as before
            self.e.constraints = {
                "time<=": self.kw["max_time"],
                "time>=": self.kw["min_time"],
            }
            download_url = self.e.get_download_url(response="csvp")

        elif self.e.protocol == "griddap":
            # the search terms that can be input for tabledap do not work for griddap
            # in erddapy currently. Instead, put together an opendap link and then
            # narrow the dataset with xarray.
            # get opendap link
            download_url = self.e.get_download_url(response="opendap")

        # add erddap server name
        return {dataset_id: [self.e.server, download_url] + items + [self.variables]}

    @property
    def meta(self):
        """Rearrange the individual metadata into a dataframe.

        Notes
        -----
        This should exclude duplicate entries.
        """

        if not hasattr(self, "_meta"):

            if self.parallel:

                # get metadata for datasets
                # run in parallel to save time
                num_cores = multiprocessing.cpu_count()
                downloads = Parallel(n_jobs=num_cores)(
                    delayed(self.meta_by_dataset)(dataset_id)
                    for dataset_id in self.dataset_ids
                )

            else:

                downloads = []
                for dataset_id in self.dataset_ids:
                    downloads.append(self.meta_by_dataset(dataset_id))

            # make dict from individual dicts
            from collections import ChainMap

            meta = dict(ChainMap(*downloads))

            # Make dataframe of metadata
            # variable names are the column names for the dataframe
            self._meta = pd.DataFrame.from_dict(
                meta,
                orient="index",
                columns=["database", "download_url"]
                + self.columns
                + ["variable names"],
            )

        return self._meta

    def data_by_dataset(self, dataset_id):
        """Return the data for a single dataset_id.

        Returns
        -------
        A tuple of (dataset_id, data), where data type is a pandas DataFrame

        Notes
        -----
        Data is read into memory.
        """

        download_url = self.meta.loc[dataset_id, "download_url"]
        # data variables in ds that are not the variables we searched for
        #         varnames = self.meta.loc[dataset_id, 'variable names']

        if self.e.protocol == "tabledap":

            try:

                # fetch metadata if not already present
                # found download_url from metadata and use
                dd = pd.read_csv(download_url, index_col=0, parse_dates=True)

                # Drop cols and rows that are only NaNs.
                dd = dd.dropna(axis="index", how="all").dropna(
                    axis="columns", how="all"
                )

                if self.variables is not None:
                    # check to see if there is any actual data
                    # this is a bit convoluted because the column names are the variable names
                    # plus units so can't match 1 to 1.
                    datacols = (
                        0  # number of columns that represent data instead of metadata
                    )
                    for col in dd.columns:
                        datacols += [
                            varname in col for varname in self.variables
                        ].count(True)
                    # if no datacols, we can skip this one.
                    if datacols == 0:
                        dd = None

            except Exception as e:
                logger.exception(e)
                logger.warning("no data to be read in for %s" % dataset_id)
                dd = None

        elif self.e.protocol == "griddap":

            try:
                dd = xr.open_dataset(download_url, chunks="auto").sel(
                    time=slice(self.kw["min_time"], self.kw["max_time"])
                )

                if ("min_lat" in self.kw) and ("max_lat" in self.kw):
                    dd = dd.sel(latitude=slice(self.kw["min_lat"], self.kw["max_lat"]))

                if ("min_lon" in self.kw) and ("max_lon" in self.kw):
                    dd = dd.sel(longitude=slice(self.kw["min_lon"], self.kw["max_lon"]))

                # use variable names to drop other variables (should. Ido this?)
                if self.variables is not None:
                    l = set(dd.data_vars) - set(self.variables)
                    dd = dd.drop_vars(l)

            except Exception as e:
                logger.exception(e)
                logger.warning("no data to be read in for %s" % dataset_id)
                dd = None

        return (dataset_id, dd)

    # @property
    def data(self):
        """Read in data for all dataset_ids.

        Returns
        -------
        A dictionary with keys of the dataset_ids and values the data of type pandas DataFrame.

        Notes
        -----
        This is either done in parallel with the `multiprocessing` library or
        in serial.
        """

        # if not hasattr(self, '_data'):

        if self.parallel:
            num_cores = multiprocessing.cpu_count()
            downloads = Parallel(n_jobs=num_cores)(
                delayed(self.data_by_dataset)(dataset_id)
                for dataset_id in self.dataset_ids
            )
        else:
            downloads = []
            for dataset_id in self.dataset_ids:
                downloads.append(self.data_by_dataset(dataset_id))

        #             if downloads is not None:
        dds = {dataset_id: dd for (dataset_id, dd) in downloads}
        #             else:
        #                 dds = None

        return dds
        # self._data = dds

        # return self._data

    def count(self, url):
        """Small helper function to count len(results) at url."""
        try:
            return len(pd.read_csv(url))
        except:
            return np.nan

    def all_variables(self):
        """Return a DataFrame of allowed variable names.

        Returns
        -------
        DataFrame of variable names and count of how many times they are present in the database.

        Notes
        -----
        This list is specific to the given ERDDAP server. If you are using
        an user-input server, it will have its own `known_server` name and upon
        running this function the first time, you should get a variable list for
        that server.

        Examples
        --------
        >>> import ocean_data_gateway as odg
        >>> odg.erddap.ErddapReader(known_server='ioos').all_variables()
                                           count
        variable
        air_pressure                        4028
        air_pressure_10011met_a                2
        air_pressure_10311ahlm_a               2
        air_pressure_10311ahlm_a_qc_agg        1
        air_pressure_10311ahlm_a_qc_tests      1
        ...                                  ...
        wind_speed_windbird_qc_agg             1
        wind_speed_windbird_qc_tests           1
        wind_to_direction                     55
        wmo_id                               954
        z                                  37377

        Or for a different `known_server`:

        >>> odg.erddap.ErddapReader(known_server='coastwatch').all_variables()
                      count
        variable
        abund_m3          2
        ac_line           1
        ac_sta            1
        adg_412           8
        adg_412_bias      8
        ...             ...
        yeardeployed      1
        yield             1
        z                 3
        z_mean            2
        zlev              6
        """

        path_name_counts = odg.variables_path.joinpath(
            f"erddap_variable_list_{self.known_server}.csv"
        )

        if path_name_counts.is_file():
            return pd.read_csv(path_name_counts, index_col="variable")
        else:
            print(
                "Please wait while the list of available variables is made. This only happens once but could take 10 minutes."
            )
            # This took 10 min running in parallel for ioos
            # 2 min for coastwatch
            url = f"{self.e.server}/categorize/variableName/index.csv?page=1&itemsPerPage=100000"
            df = pd.read_csv(url)
            if not self.parallel:
                counts = []
                for url in df.URL:
                    counts.append(self.count(url))
            else:
                num_cores = multiprocessing.cpu_count()
                counts = Parallel(n_jobs=num_cores)(
                    delayed(self.count)(url) for url in df.URL
                )
            dfnew = pd.DataFrame()
            dfnew["variable"] = df["Category"]
            dfnew["count"] = counts
            dfnew = dfnew.set_index("variable")
            # remove nans
            if (dfnew.isnull().sum() > 0).values:
                dfnew = dfnew[~dfnew.isnull().values].astype(int)
            dfnew.to_csv(path_name_counts)

        return dfnew

    def search_variables(self, variables):
        """Find valid variables names to use.

        Parameters
        ----------
        variables: string, list
            String or list of strings to use in regex search to find valid
            variable names.

        Returns
        -------
        DataFrame of variable names and count of how many times they are present in the database, sorted by count.

        Notes
        -----
        This list is only specific to the ERDDAP server.

        Examples
        --------

        Search for variables that contain the substring 'sal':

        >>> odg.erddap.ErddapReader(known_server='ioos').search_variables('sal')
                                                        count
        variable
        salinity                                          954
        salinity_qc                                       954
        sea_water_practical_salinity                      778
        soil_salinity_qc_agg                              622
        soil_salinity                                     622
        ...                                               ...
        sea_water_practical_salinity_4161sc_a_qc_tests      1
        sea_water_practical_salinity_6754mc_a_qc_tests      1
        sea_water_practical_salinity_6754mc_a_qc_agg        1
        sea_water_practical_salinity_4161sc_a_qc_agg        1
        sea_water_practical_salinity_10091sc_a              1

        Find available variables sorted by count:

        >>>  odg.erddap.ErddapReader(known_server='ioos').search_variables('')
                                                            count
        variable
        time                                                38331
        longitude                                           38331
        latitude                                            38331
        z                                                   37377
        station                                             37377
        ...                                                   ...
        sea_surface_wave_from_direction_elw11a3t01wv_qc...      1
        sea_surface_wave_from_direction_elw11b2t01wv            1
        sea_surface_wave_from_direction_elw11b2t01wv_qc...      1
        sea_surface_wave_from_direction_elw11b2t01wv_qc...      1
        sea_water_pressure_7263arc_a                            1
        """

        if not isinstance(variables, list):
            variables = [variables]

        # set up search for input variables
        search = f"(?i)"
        for variable in variables:
            search += f".*{variable}|"
        search = search.strip("|")

        r = re.compile(search)

        # just get the variable names
        df = self.all_variables()
        parameters = df.index

        matches = list(filter(r.match, parameters))

        # return parameters that match input variable strings
        return df.loc[matches].sort_values("count", ascending=False)

    def check_variables(self, variables, verbose=False):
        """Checks variables for presence in database list.

        Parameters
        ----------
        variables: string, list
            String or list of strings to compare against list of valid
            variable names.
        verbose: boolean, optional
            Print message if variables are matches instead of passing silently.

        Returns
        -------
        Nothing is returned. However, there are two types of behavior:

        if variables is not a valid variable name(s), an AssertionError is
          raised and `search_variables(variables)` is run on your behalf to
          suggest valid variable names to use.
        if variables is a valid variable name(s), nothing happens.

        Notes
        -----
        This list is specific to the ERDDAP server being used.

        Examples
        --------
        Check if the variable name 'sal' is valid:

        >>> odg.erddap.ErddapReader(known_server='ioos').check_variables('sal')
        AssertionError                            Traceback (most recent call last)
        <ipython-input-13-f8082c9bfafa> in <module>
        ----> 1 odg.erddap.ErddapReader(known_server='ioos').check_variables('sal')
        ~/projects/ocean_data_gateway/ocean_data_gateway/readers/erddap.py in check_variables(self, variables, verbose)
            572         salinity_qc                                       954
            573         sea_water_practical_salinity                      778
        --> 574         soil_salinity_qc_agg                              622
            575         soil_salinity                                     622
            576         ...                                               ...
        AssertionError: The input variables are not exact matches to ok variables for known_server ioos.
        Check all parameter group values with `ErddapReader().all_variables()`
        or search parameter group values with `ErddapReader().search_variables(['sal'])`.
         Try some of the following variables:
                                                        count
        variable
        salinity                                          954
        salinity_qc                                       954
        sea_water_practical_salinity                      778
        soil_salinity_qc_agg                              622
        soil_salinity                                     622
        ...                                               ...
        sea_water_practical_salinity_4161sc_a_qc_tests      1
        sea_water_practical_salinity_6754mc_a_qc_tests      1
        sea_water_practical_salinity_6754mc_a_qc_agg        1
        sea_water_practical_salinity_4161sc_a_qc_agg        1
        sea_water_practical_salinity_10091sc_a              1

        Check if the variable name 'salinity' is valid:

        >>> odg.erddap.ErddapReader(known_server='ioos').check_variables('salinity')

        """

        if not isinstance(variables, list):
            variables = [variables]

        parameters = list(self.all_variables().index)

        # for a variable to exactly match a parameter
        # this should equal 1
        count = []
        for variable in variables:
            count += [parameters.count(variable)]

        condition = np.allclose(count, 1)

        assertion = f"The input variables are not exact matches to ok variables for known_server {self.known_server}. \
                     \nCheck all parameter group values with `ErddapReader().all_variables()` \
                     \nor search parameter group values with `ErddapReader().search_variables({variables})`.\
                     \n\n Try some of the following variables:\n{str(self.search_variables(variables))}"  # \
        assert condition, assertion

        if condition and verbose:
            print("all variables are matches!")


# Search for stations by region
class region(ErddapReader):
    """Inherits from ErddapReader to search over a region of space and time.

    Attributes
    ----------
    kw: dict
      Contains space and time search constraints: `min_lon`, `max_lon`,
      `min_lat`, `max_lat`, `min_time`, `max_time`.
    variables: string or list
      Variable names if you want to limit the search to those. The variable name or names must be from the list available in `all_variables()` for the specific ERDDAP server and pass the check in `check_variables()`.
    approach: string
        approach is defined as 'region' for this class.
    """

    def __init__(self, kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Can contain arguments to pass onto the base ErddapReader class
            (known_server, protocol, server, parallel). The dict entries to
            initialize this class are:

            * kw: dict
              Contains space and time search constraints: `min_lon`, `max_lon`,
              `min_lat`, `max_lat`, `min_time`, `max_time`.
            * variables: string or list, optional
              Variable names if you want to limit the search to those. The variable name or names must be from the list available in `all_variables()` for the specific ERDDAP server and pass the check in `check_variables()`.
        """
        assert isinstance(kwargs, dict), "input arguments as dictionary"
        er_kwargs = {
            "known_server": kwargs.get("known_server", "ioos"),
            "protocol": kwargs.get("protocol", None),
            "server": kwargs.get("server", None),
            "parallel": kwargs.get("parallel", True),
        }
        ErddapReader.__init__(self, **er_kwargs)

        kw = kwargs["kw"]
        variables = kwargs.get("variables", None)

        self.approach = "region"

        self._stations = None

        # run checks for KW
        # check for lon/lat values and time
        self.kw = kw

        if (variables is not None) and (not isinstance(variables, list)):
            variables = [variables]

        # make sure variables are on parameter list
        if variables is not None:
            self.check_variables(variables)
        self.variables = variables


class stations(ErddapReader):
    """Inherits from ErddapReader to search for 1+ stations or dataset_ids.

    Attributes
    ----------
    kw: dict, optional
      Contains space and time search constraints: `min_time`, `max_time`.
    variables: None
        variables is None for this class since we read search by dataset_id or
        station name.
    approach: string
        approach is defined as 'stations' for this class.
    """

    def __init__(self, kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Can contain arguments to pass onto the base ErddapReader class
            (known_server, protocol, server, parallel). The dict entries to
            initialize this class are:
            * kw: dict, optional
              Contains time search constraints: `min_time`, `max_time`.
              If not input, all time will be used.
            * dataset_ids: string, list, optional
              Use this option if you know the exact dataset_ids for the data
              you want. These need to be the dataset_ids corresponding to the
              databases that are being searched, so in this case they need to be
               the ERDDAP server's dataset_ids. If you know station names but
              not the specific database uuids, input the names as "stations"
              instead.
            * stations: string, list, optional
              Input station names as they might be commonly known and therefore
              can be searched for as a query term. The station names can be
              input as something like "TABS B" or "8771972" and has pretty good
              success.

        Notes
        -----
        The known_server needs to match the station name or dataset_id you are
        searching for.
        """
        assert isinstance(kwargs, dict), "input arguments as dictionary"
        er_kwargs = {
            "known_server": kwargs.get("known_server", "ioos"),
            "protocol": kwargs.get("protocol", None),
            "server": kwargs.get("server", None),
            "parallel": kwargs.get("parallel", True),
        }
        ErddapReader.__init__(self, **er_kwargs)

        kw = kwargs.get("kw", None)
        dataset_ids = kwargs.get("dataset_ids", None)
        stations = kwargs.get("stations", [])

        self.approach = "stations"

        # we want all the data associated with stations
        self.variables = None

        if dataset_ids is not None:
            if not isinstance(dataset_ids, list):
                dataset_ids = [dataset_ids]
            self._dataset_ids = dataset_ids

        if not stations == []:
            if not isinstance(stations, list):
                stations = [stations]
            self._stations = stations
            self.dataset_ids
        else:
            self._stations = stations

        # CHECK FOR KW VALUES AS TIMES
        if kw is None:
            kw = {"min_time": "1900-01-01", "max_time": "2100-12-31"}

        self.kw = kw

"""
Reader for ERDDAP servers.
"""

import logging
import multiprocessing
import urllib.parse

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr

from erddapy import ERDDAP
from joblib import Parallel, delayed

import ocean_data_gateway as odg

from ocean_data_gateway import Reader


logger = logging.getLogger(__name__)

# this can be queried with
# search.ErddapReader.reader
reader = "erddap"


class ErddapReader(Reader):
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

        # hard wire this for now
        filetype = "netcdf"

        # either select a known server or input protocol and server string
        if known_server == "ioos":
            protocol = "tabledap"
            server = "http://erddap.sensors.ioos.us/erddap"
            filetype = "netcdf"  # other option: "csv"
        elif known_server == "coastwatch":
            protocol = "griddap"
            server = "http://coastwatch.pfeg.noaa.gov/erddap"
            filetype = "netcdf"  # other option: "csv"
        elif known_server is not None:
            statement = (
                "either select a known server or input protocol and server string"
            )
            assert (protocol is not None) & (server is not None), statement
        else:
            known_server = urllib.parse.urlparse(server).netloc
            # known_server = server.strip("/erddap").strip("http://").replace(".", "_")
            statement = (
                "either select a known server or input protocol and server string"
            )
            assert (protocol is not None) & (server is not None), statement

        self.known_server = known_server
        self.e = ERDDAP(server=server)
        self.e.protocol = protocol
        self.e.server = server
        self.filetype = filetype

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
        self.store = dict()

    def __getitem__(self, key):
        """Redefinition of dict-like behavior.

        This enables user to use syntax `reader[dataset_id]` to read in and
        save dataset into the object.

        Parameters
        ----------
        key: str
            dataset_id for a dataset that is available in the search/reader
            object.

        Returns
        -------
        xarray Dataset of the data associated with key
        """

        returned_data = self.data_by_dataset(key)
        # returned_data = self._return_data(key)
        self.__setitem__(key, returned_data)
        return returned_data

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
                # first try as dataset_id then do as station name
                dataset_id = [
                    dataset_id
                    for dataset_id in df["Dataset ID"]
                    if station.lower()
                    in [dataset_id.lower()] + dataset_id.lower().split("_")
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
        The dataset_ids are found by querying the metadata through the ERDDAP server.

        The number of dataset_ids can change if a variable is removed from the
        list of variables and this is rerun.
        """

        if not hasattr(self, "_dataset_ids") or (
            self.variables and (len(self.variables) != self.num_variables)
        ):

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

                # remove None from list
                dataset_ids = [i for i in dataset_ids if i]

                # In this case return all dataset_ids so they match 1-1 with
                # the input station list.
                self._dataset_ids = dataset_ids

            else:
                logger.warning(
                    "Neither stations nor region approach were used in function dataset_ids."
                )

            # update number of variables
            if self.variables:
                self.num_variables = len(self.variables)

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

        # include download link ##
        self.e.dataset_id = dataset_id
        if self.e.protocol == "tabledap":
            # set the same time restraints as before
            self.e.constraints = {
                "time<=": self.kw["max_time"],
                "time>=": self.kw["min_time"],
            }
            if self.filetype == "csv":
                download_url = self.e.get_download_url(response="csvp")
            elif self.filetype == "netcdf":
                download_url = self.e.get_download_url(response="ncCf")

        elif self.e.protocol == "griddap":
            # the search terms that can be input for tabledap do not work for griddap
            # in erddapy currently. Instead, put together an opendap link and then
            # narrow the dataset with xarray.
            # get opendap link
            download_url = self.e.get_download_url(response="opendap")

        # check if "prediction" is present in metadata, esp in case of NOAA
        # model predictions
        is_prediction = "Prediction" in " ".join(
            list(info["Value"].replace(np.nan, None).values)
        )

        # add erddap server name
        return {
            dataset_id: [self.e.server, download_url, info_url, is_prediction]
            + items
            + [self.variables]
        }

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
                columns=["database", "download_url", "info_url", "is_prediction"]
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

        if self.filetype == "csv":
            # if self.e.protocol == "tabledap":
            try:
                # fetch metadata if not already present
                # found download_url from metadata and use
                self.e.dataset_id = dataset_id
                # dataset_vars gives a list of the variables in the dataset
                dataset_vars = (
                    self.meta.loc[dataset_id]["defaultDataQuery"]
                    .split("&")[0]
                    .split(",")
                )
                # vars_present gives the variables in self.variables
                # that are actually in the dataset
                vars_present = []
                for selfvariable in self.variables:
                    vp = [var for var in dataset_vars if var == selfvariable]
                    if len(vp) > 0:
                        vars_present.append(vp[0])
                # If any variables are not present, this doesn't work.
                if self.variables is not None:
                    self.e.variables = [
                        "time",
                        "longitude",
                        "latitude",
                        "station",
                    ] + vars_present
                dd = self.e.to_pandas(response="csvp", index_col=0, parse_dates=True)
                # dd = self.e.to_pandas(response='csv', header=[0, 1],
                #                       index_col=0, parse_dates=True)
                # dd = pd.read_csv(
                #     download_url, header=[0, 1], index_col=0, parse_dates=True
                # )

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

        elif self.filetype == "netcdf":
            # elif self.e.protocol == "griddap":

            if self.e.protocol == "tabledap":

                try:
                    # assume I don't need to narrow in space since time series (tabledap)
                    self.e.dataset_id = dataset_id
                    dd = self.e.to_xarray()
                    # dd = xr.open_dataset(download_url, chunks="auto")
                    dd = dd.swap_dims({"obs": dd.cf["time"].name})
                    dd = dd.sortby(dd.cf["time"], ascending=True)
                    dd = dd.cf.sel(T=slice(self.kw["min_time"], self.kw["max_time"]))
                    # dd = dd.set_coords(
                    #     [dd.cf["longitude"].name, dd.cf["latitude"].name]
                    # )

                    # use variable names to drop other variables (should. Ido this?)
                    if self.variables is not None:
                        # I don't think this is true with new approach
                        # # ERDDAP prepends variables with 's.' in netcdf files,
                        # # so include those with variables
                        # erd_vars = [f's.{var}' for var in self.variables]
                        # var_list = set(dd.data_vars) - (set(self.variables) | set(erd_vars))
                        var_list = set(dd.data_vars) - set(self.variables)
                        dd = dd.drop_vars(var_list)

                    # the lon/lat are on the 'timeseries' singleton dimension
                    # but the data_var variable was not, which messed up
                    # cf-xarray. When longitude and latitude are not on a
                    # dimension shared with a variable, the variable can't be
                    # called with cf-xarray. e.g. dd.cf['ssh'] won't work.
                    if "timeseries" in dd.dims:
                        for data_var in dd.data_vars:
                            if "timeseries" not in dd[data_var].dims:
                                dd[data_var] = dd[data_var].expand_dims(
                                    dim="timeseries", axis=1
                                )

                except Exception as e:
                    logger.exception(e)
                    logger.warning("no data to be read in for %s" % dataset_id)
                    dd = None

            elif self.e.protocol == "griddap":

                try:
                    # this makes it read in the whole file which might be large
                    self.e.dataset_id = dataset_id
                    # dd = self.e.to_xarray(chunks="auto").sel(
                    #     time=slice(self.kw["min_time"], self.kw["max_time"])
                    # )
                    download_url = self.e.get_download_url(response="opendap")
                    dd = xr.open_dataset(download_url, chunks="auto").sel(
                        time=slice(self.kw["min_time"], self.kw["max_time"])
                    )

                    if ("min_lat" in self.kw) and ("max_lat" in self.kw):
                        dd = dd.sel(
                            latitude=slice(self.kw["min_lat"], self.kw["max_lat"])
                        )

                    if ("min_lon" in self.kw) and ("max_lon" in self.kw):
                        dd = dd.sel(
                            longitude=slice(self.kw["min_lon"], self.kw["max_lon"])
                        )

                    # use variable names to drop other variables (should. Ido this?)
                    if self.variables is not None:
                        vars_list = set(dd.data_vars) - set(self.variables)
                        dd = dd.drop_vars(vars_list)

                except Exception as e:
                    logger.exception(e)
                    logger.warning("no data to be read in for %s" % dataset_id)
                    dd = None

        # return (dataset_id, dd)
        return dd

    # @property
    def data(self, dataset_ids=None):
        """Read in data for some or all dataset_ids.

        NOT USED CURRENTLY

        Once data is read in for a dataset_ids, it is remembered.

        See full documentation in `utils.load_data()`.
        """

        output = odg.utils.load_data(self, dataset_ids)
        return output


# Search for stations by region
class region(ErddapReader):
    """Inherits from ErddapReader to search over a region of space and time.

    Attributes
    ----------
    kw: dict
      Contains space and time search constraints: `min_lon`, `max_lon`,
      `min_lat`, `max_lat`, `min_time`, `max_time`.
    variables: string or list
      Variable names if you want to limit the search to those. The variable name or names must be from the list available in `odg.all_variables(server)` for the specific ERDDAP server and pass the check in `odg.check_variables(server, variables)`.
    criteria: dict, str, optional
      A dictionary describing how to recognize variables by their name
      and attributes with regular expressions to be used with
      `cf-xarray`. It can be local or a URL point to a nonlocal gist.
      This is required for running QC in Gateway. For example:
      ```
      my_custom_criteria = {
        "salt": {
            "standard_name": "sea_water_salinity$|sea_water_practical_salinity$",
            "name": (?i)sal$|(?i)s.sea_water_practical_salinity$",
        },
      }
      ```
    var_def: dict, optional
      A dictionary with the same keys as criteria (criteria can have
      more) that describes QC definitions and units. It should include
      the variable units, fail_span, and suspect_span. For example:
      ```
      var_def = {
        "salt": {"units": "psu", "fail_span": [-10, 60],
                 "suspect_span": [-1, 45]},
      }
      ```
    approach: string
        approach is defined as 'region' for this class.
    num_variables: int
        Number of variables stored in self.variables. This is set initially and
        if self.variables is modified, this is updated accordingly. If
        `variables is None`, `num_variables==0`.
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
              Variable names if you want to limit the search to those. The variable name or names must be from the list available in `odg.all_variables(server)` for the specific ERDDAP server and pass the check in `odg.check_variables(server, variables)`.

              Alternatively, if the user inputs criteria, variables can be a
              list of the keys from criteria.
            * criteria: dict, optional
              A dictionary describing how to recognize variables by their name
              and attributes with regular expressions to be used with
              `cf-xarray`. It can be local or a URL point to a nonlocal gist.
              This is required for running QC in Gateway. For example:
              ```
              my_custom_criteria = {
                "salt": {
                    "standard_name": "sea_water_salinity$|sea_water_practical_salinity$",
                    "name": (?i)sal$|(?i)s.sea_water_practical_salinity$",
                },
              }
              ```
            * var_def: dict, optional
              A dictionary with the same keys as criteria (criteria can have
              more) that describes QC definitions and units. It should include
              the variable units, fail_span, and suspect_span. For example:
              ```
              var_def = {
                "salt": {"units": "psu", "fail_span": [-10, 60],
                         "suspect_span": [-1, 45]},
              }
              ```
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

        # check for custom criteria to set up cf-xarray
        if "criteria" in kwargs:
            criteria = kwargs["criteria"]
            # link to nonlocal dictionary definition
            if isinstance(criteria, str) and criteria[:4] == "http":
                criteria = odg.return_response(criteria)
            cf_xarray.set_options(custom_criteria=criteria)
            self.criteria = criteria
        else:
            self.criteria = None

        if (variables is not None) and (not isinstance(variables, list)):
            variables = [variables]

        # make sure variables are on parameter list
        if variables is not None:
            # User is using criteria and variable nickname approach
            if self.criteria and all(var in self.criteria for var in variables):
                # first translate the input variable nicknames to variable names
                # that are specific to the reader.
                variables = odg.select_variables(
                    self.e.server, self.criteria, variables
                )

            # user is inputting specific reader variable names
            else:
                odg.check_variables(self.e.server, variables)
            # record the number of variables so that a user can change it and
            # the change can be compared.
            self.num_variables = len(variables)
        else:
            self.num_variables = 0
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
            * stations: string, list, optional
              Input station names as they might be commonly known and therefore
              can be searched for as a query term. The station names can be
              input as something like "TABS B" or "8771972" and has pretty good
              success.
              Or, input the exact dataset_ids for the data you want,
              corresponding to the databases that are being searched, so in this
              case they need to be the ERDDAP server's dataset_ids.

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
        stations = kwargs.get("stations", [])

        self.approach = "stations"

        # we want all the data associated with stations
        self.variables = None

        if not stations == []:
            if not isinstance(stations, list):
                stations = [stations]
        self._stations = stations

        # CHECK FOR KW VALUES AS TIMES
        if kw is None:
            kw = {"min_time": "1900-01-01", "max_time": "2100-12-31"}

        self.kw = kw

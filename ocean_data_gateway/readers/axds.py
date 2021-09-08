"""
Reader for Axiom databases.
"""

import hashlib
import logging
import os

import cf_xarray
import fsspec
import intake
import numpy as np
import pandas as pd
import requests
import xarray as xr

import ocean_data_gateway as odg

from ocean_data_gateway import Reader


logger = logging.getLogger(__name__)

# this can be queried with
# search.AxdsReader.reader
reader = "axds"


class AxdsReader(Reader):
    """
    This class searches Axiom databases for types `platforms2`, which
    are like gliders, and `layer_group`, which are like grids and models.

    Attributes
    ----------
    parallel: boolean
        If True, run with simple parallelization using `multiprocessing`.
        If False, run serially.
    catalog_name: string
        Input catalog path if you want to use an existing catalog.
    axds_type: string
        Which Axiom database type to search for.
        * "platform2" (default): gliders, drifters; result in pandas DataFrames
        * "layer_group": grids, model output; result in xarray Datasets
    url_search_base: string
        Base string of search url
    url_docs_base: string
        Base string of url for a known dataset_id
    search_headers: dict
        Required for reading in the request
    url_axds_type: string
        Url for the given `axds_type`.
    name: string
        f'axds_{axds_type}' so 'axds_platform2' or 'axds_layer_group'
    reader: string
        Reader name: AxdsReader
    """

    def __init__(
        self, parallel=True, catalog_name=None, axds_type="platform2", filetype="netcdf"
    ):
        """
        Parameters
        ----------
        parallel: boolean, optional
            If True, run with simple parallelization using `multiprocessing`.
            If False, run serially.
        catalog_name: string, optional
            Input catalog path if you want to use an existing catalog.
        axds_type: string, optional
            Which Axiom database type to search for.
            * "platform2" (default): gliders, drifters; result in pandas DataFrames
            * "layer_group": grids, model output; result in xarray Datasets
        """

        self.parallel = parallel

        # search Axiom database, version 2
        self.url_search_base = "https://search.axds.co/v2/search?portalId=-1&page=1&pageSize=10000&verbose=true"
        self.url_docs_base = "https://search.axds.co/v2/docs?verbose=true"

        # this is the json being returned from the request
        self.search_headers = {"Accept": "application/json"}

        self.approach = None

        if catalog_name is None:
            name = f"{pd.Timestamp.now().isoformat()}"
            hash_name = hashlib.sha256(name.encode()).hexdigest()[:7]
            self.catalog_name = odg.catalogs_path.joinpath(f"catalog_{hash_name}.yml")
        else:
            self.catalog_name = catalog_name
            # if catalog_name already exists, read it in to save time
            self.catalog

        # can be 'platform2' or 'layer_group'
        assert axds_type in [
            "platform2",
            "layer_group",
        ], 'variable `axds_type` must be "platform2" or "layer_group"'
        self.axds_type = axds_type

        self.url_axds_type = f"{self.url_search_base}&type={self.axds_type}"
        self.name = f"axds_{axds_type}"
        self.reader = "AxdsReader"

        if self.axds_type == "platform2":
            self.data_type = "csv"
        elif self.axds_type == "layer_group":
            self.data_type = "nc"

        # name
        self.name = f"axds_{axds_type}"

        self.reader = "AxdsReader"

        self.filetype = filetype
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

    def url_query(self, query):
        """url modification to add query field.

        Parameters
        ----------
        query: string
            String to query for. Can be multiple words.

        Returns
        -------
        Modification for url to add query field.
        """
        return f"&query={query}"

    def url_variable(self, variable):
        """url modification to add variable search.

        Parameters
        ----------
        variable: string
            String to search for.

        Returns
        -------
        Modification for url to add variable search.

        Notes
        -----
        This variable search is specifically by parameter group and
        only works for `axds_type='platform2'`.
        For `axds_type='layer_group'`, use `url_query` with the variable name.
        """
        return f"&tag=Parameter+Group:{variable}"

    def url_region(self):
        """url modification to add spatial search box.

        Returns
        -------
        Modification for url to add lon/lat filtering.

        Notes
        -----
        Uses the `kw` dictionary already stored in the class object
        to access the spatial limits of the box.
        """
        url_add_box = (
            f'&geom={{"type":"Polygon","coordinates":[[[{self.kw["min_lon"]},{self.kw["min_lat"]}],'
            + f'[{self.kw["max_lon"]},{self.kw["min_lat"]}],'
            + f'[{self.kw["max_lon"]},{self.kw["max_lat"]}],'
            + f'[{self.kw["min_lon"]},{self.kw["max_lat"]}],'
            + f'[{self.kw["min_lon"]},{self.kw["min_lat"]}]]]}}'
        )
        return f"{url_add_box}"

    def url_time(self):
        """url modification to add time filtering.

        Returns
        -------
        Modification for url to add time filtering.

        Notes
        -----
        Uses the `kw` dictionary already stored in the class object
        to access the time limits of the search.
        """
        # convert input datetime to seconds since 1970
        startDateTime = (
            pd.Timestamp(self.kw["min_time"]).tz_localize("UTC")
            - pd.Timestamp("1970-01-01 00:00").tz_localize("UTC")
        ) // pd.Timedelta("1s")
        endDateTime = (
            pd.Timestamp(self.kw["max_time"]).tz_localize("UTC")
            - pd.Timestamp("1970-01-01 00:00").tz_localize("UTC")
        ) // pd.Timedelta("1s")

        # search by time
        url_add_time = f"&startDateTime={startDateTime}&endDateTime={endDateTime}"

        return f"{url_add_time}"

    def url_dataset_id(self, dataset_id):
        """url modification to search for known dataset_id.

        Parameters
        ----------
        dataset_id: string
            String of dataset_id to exactly match.

        Returns
        -------
        Modification for url to search for dataset_id.
        """
        return f"&id={dataset_id}"

    def url_builder(
        self,
        url_base,
        dataset_id=None,
        add_region=False,
        add_time=False,
        variable=None,
        query=None,
    ):
        """Build an individual search url.

        Parameters
        ----------
        url_base: string
            There are 2 possible bases for the url:
            * self.url_axds_type, for searching
            * self.url_docs_base, for selecting known dataset by dataset_id
        dataset_id: string, optional
            dataset_id of station, if known.
        add_region: boolean, optional
            True to filter the search by lon/lat box. Requires self.kw
            that contains keys `min_lon`, `max_lon`, `min_lat`, `max_lat`.
        add_time: boolean, optional
            True to filter the search by time range. Requires self.kw
            that contains keys `min_time` and `max_time`.
        variable: string, optional
            String of variable description to filter by, if desired.
            If `axds_type=='platform2'`, find the variable name options with
            class function `odg.all_variables('axds')`, search for variable names by
            string with `odg.search_variables('axds', variables)`, and check your variable list with
            `check_variables('axds', variables)`.
            If `axds_type=='layer_group'`, there is no official variable list
            and you can instead just put in a basic variable name and hope the
            search works.
        query: string, optional
            This could be any search query you want, but it is used in the code
            to search for station names (not dataset_ids).

        Returns
        -------
        Url for search.
        """
        url = url_base
        if dataset_id is not None:
            url += self.url_dataset_id(dataset_id)
        if add_time:
            url += self.url_time()
        if variable is not None:
            if self.axds_type == "platform2":
                url += self.url_variable(variable)
            elif self.axds_type == "layer_group":
                url += self.url_query(variable)
        if add_region:
            url += self.url_region()
        if query is not None:
            url += self.url_query(query)

        return url

    @property
    def urls(self):
        """Return a list of search urls.

        Notes
        -----
        Use this through the class methods `region` or `stations` to put
        together the search urls to represent the basic reader setup.
        """

        assert (
            self.approach is not None
        ), "Use this property through class method `region` or `stations`"

        if not hasattr(self, "_urls"):

            if self.approach == "region":
                urls = []
                if self.variables is not None:
                    for variable in self.variables:
                        urls.append(
                            self.url_builder(
                                self.url_axds_type,
                                variable=variable,
                                add_time=True,
                                add_region=True,
                            )
                        )
                else:
                    urls.append(
                        self.url_builder(
                            self.url_axds_type, add_time=True, add_region=True
                        )
                    )

            elif self.approach == "stations":
                urls = []
                # check station names as both queries and as exact names
                if len(self._stations) > 0:
                    for station in self._stations:
                        urls.append(self.url_builder(self.url_axds_type, query=station))
                        urls.append(
                            self.url_builder(self.url_docs_base, dataset_id=station)
                        )

            self._urls = urls

        return self._urls

    @property
    def search_results(self):
        """Loop over self.urls to read in search results.

        Notes
        -----
        The logic removes duplicate searches.
        This returns a dict of the datasets from the search results with the
        key of each entry being the dataset_id. For

        * `self.axds_type == "platform2"`: dataset_id is the uuid
        * `self.axds_type == "layer_group"`: dataset_id is the module_uuid since multiple layer_groups can be linked under one module_uuid
        """

        if not hasattr(self, "_search_results"):

            # loop over urls
            search_results = []
            for url in self.urls:
                # first make sure is legitimate web address
                if requests.get(url).status_code == 200:
                    res = requests.get(url, headers=self.search_headers).json()
                    # get different returns for an id docs grab vs. generic search
                    #                 if isinstance(res, list):
                    #                     res = res[0]
                    if isinstance(res, dict):
                        res = res["results"]
                    search_results.extend(res)
            # change search_results to a dictionary to remove
            # duplicate dataset_ids
            search_results_dict = {}
            for search_result in search_results:
                if self.axds_type == "platform2":
                    search_results_dict[search_result["uuid"]] = search_result
                #                     search_results_dict[search_result['data']['uuid']] = search_result
                if self.axds_type == "layer_group":
                    # this is in the case that our search results are for a layer_group
                    if ("module_uuid" in search_result["data"]) and (
                        search_result["type"] == "layer_group"
                    ):
                        # switch to module search results instead of layer_group results
                        module_uuid = search_result["data"]["module_uuid"]
                    # this is the case that our searcb results are for a module
                    elif search_result["type"] == "module":
                        module_uuid = search_result["data"]["uuid"]

                    # don't repeat unnecessarily, if module_uuid has already
                    # been included.
                    if module_uuid in search_results_dict.keys():
                        continue
                    else:
                        url_module = self.url_builder(
                            self.url_docs_base, dataset_id=module_uuid
                        )
                        search_results_dict[module_uuid] = requests.get(
                            url_module, headers=self.search_headers
                        ).json()[0]

            condition = search_results_dict == {}
            assertion = f"No datasets fit the input criteria of kw={self.kw} and variables={self.variables}"
            #             assert condition, assertion
            if condition:
                logger.warning(assertion)
                # self._dataset_ids = []

            # DON'T SAVE THIS LATER, JUST FOR DEBUGGING
            self._search_results = search_results_dict

        #             self._dataset_ids = list(search_results_dict.keys())
        return self._search_results

    def write_catalog_layer_group_entry(
        self, dataset, dataset_id, urlpath, layer_groups
    ):
        """Write part of catalog in case of layer_group.

        Notes
        -----
        This is used to manage the logic for `axds_type='layer_group'` in which
        the module is being linked to the set of layer_groups.
        """

        try:
            model_slug = dataset["data"]["model"]["slug"]
        except:
            model_slug = ""

        # these are from the module
        try:
            label = dataset["label"].replace(":", "-")
        except:
            label = dataset["data"]["short_description"]

        geospatial_lat_min, geospatial_lat_max = (
            dataset["data"]["min_lat"],
            dataset["data"]["max_lat"],
        )
        geospatial_lon_min, geospatial_lon_max = (
            dataset["data"]["min_lng"],
            dataset["data"]["max_lng"],
        )

        # set up lines
        file_intake = intake.open_opendap(
            urlpath, engine="netcdf4", xarray_kwargs=dict()
        )
        file_intake.description = label
        file_intake.engine = "netcdf4"
        metadata = {
            "urlpath": urlpath,
            "variables": list(layer_groups.values()),
            "layer_group_uuids": list(layer_groups.keys()),
            "model_slug": model_slug,
            "geospatial_lon_min": geospatial_lon_min,
            "geospatial_lat_min": geospatial_lat_min,
            "geospatial_lon_max": geospatial_lon_max,
            "geospatial_lat_max": geospatial_lat_max,
            "time_coverage_start": dataset["start_date_time"],
            "time_coverage_end": dataset["end_date_time"],
        }
        file_intake.metadata = metadata
        file_intake.name = dataset_id
        lines = file_intake.yaml().strip("sources:")

        return lines

    def write_catalog(self):
        """Write catalog file."""

        # if the catalog already exists, don't do this
        if os.path.exists(self.catalog_name):
            return

        else:

            f = open(self.catalog_name, "w")

            if self.axds_type == "platform2":
                lines = "sources:\n"

                for dataset_id, dataset in self.search_results.items():
                    if self.filetype == "csv":
                        urlpath = dataset["source"]["files"]["data.csv.gz"]["url"]
                        file_intake = intake.open_csv(
                            urlpath, csv_kwargs=dict(parse_dates=["time"])
                        )
                    elif self.filetype == "netcdf":
                        key = [
                            key
                            for key in dataset["source"]["files"].keys()
                            if ".nc" in key
                        ][0]
                        urlpath = dataset["source"]["files"][key]["url"]
                        file_intake = intake.open_netcdf(
                            urlpath
                        )  # , xarray_kwargs=dict(parse_dates=['time']))
                    # to get all metadata
                    # source = intake.open_textfiles(meta_url, decoder=json.loads)
                    # source.metadata = source.read()[0]
                    meta_url = dataset["source"]["files"]["meta.json"]["url"]
                    meta_url = meta_url.replace(" ", "%20")
                    attributes = pd.read_json(meta_url)["attributes"]
                    file_intake.description = attributes["summary"]
                    metadata = {
                        "urlpath": urlpath,
                        "meta_url": meta_url,
                        "platform_category": attributes["platform_category"],
                        "geospatial_lon_min": attributes["geospatial_lon_min"],
                        "geospatial_lat_min": attributes["geospatial_lat_min"],
                        "geospatial_lon_max": attributes["geospatial_lon_max"],
                        "geospatial_lat_max": attributes["geospatial_lat_max"],
                        "source_id": attributes["packrat_source_id"],
                        "packrat_uuid": attributes["packrat_uuid"],
                        "time_coverage_start": attributes["time_coverage_start"],
                        "time_coverage_end": attributes["time_coverage_end"],
                    }
                    file_intake.metadata = metadata
                    file_intake.name = attributes["packrat_uuid"]
                    lines += file_intake.yaml().strip("sources:")

            elif self.axds_type == "layer_group":
                lines = """
plugins:
  source:
    - module: intake_xarray
sources:
"""
                # catalog entries are by module uuid and unique to opendap urls
                # dataset_ids are module uuids
                for dataset_id, dataset in self.search_results.items():

                    # layer_groups associated with module
                    layer_groups = dataset["data"]["layer_group_info"]

                    # get search results for layer_groups
                    urlpaths = []
                    for layer_group_uuid in layer_groups.keys():
                        url_layer_group = self.url_builder(
                            self.url_docs_base, dataset_id=layer_group_uuid
                        )
                        search_results_lg = requests.get(
                            url_layer_group, headers=self.search_headers
                        ).json()[0]

                        if "OPENDAP" in search_results_lg["data"]["access_methods"]:
                            url = search_results_lg["source"]["layers"][0][
                                "thredds_opendap_url"
                            ]
                            if ".html" in url:
                                url = url.replace(".html", "")
                            urlpaths.append(url)
                        else:
                            urlpaths.append("")
                            logger.warning(
                                f"no opendap url for module: module uuid {dataset_id}, layer_group uuid {layer_group_uuid}"
                            )
                            continue

                    # there may be different urls for different layer_groups
                    # in which case associate the layer_group uuid with the dataset
                    # since the module uuid wouldn't be unique
                    # if there were no urlpaths for any of the layer_groups,
                    # urlpaths is like ['', '', '', '', '', '', '', '']
                    if len(set(urlpaths)) > 1:
                        logger.warning(
                            f"there are multiple urls for module: module uuid {dataset_id}. urls: {set(urlpaths)}"
                        )
                        for urlpath, layer_group_uuid in zip(
                            urlpaths, layer_groups.keys()
                        ):
                            lines += self.write_catalog_layer_group_entry(
                                dataset, layer_group_uuid, urlpath, layer_groups
                            )

                    # check for when no urlpaths, don't save entry
                    # if not opendap accessible
                    elif set(urlpaths) == {""}:
                        logger.warning(
                            f"no opendap url for module: module uuid {dataset_id} for any of its layer_groups. Do not include entry in catalog."
                        )
                        continue

                    else:
                        urlpath = list(set(urlpaths))[0]
                        # use module uuid
                        lines += self.write_catalog_layer_group_entry(
                            dataset, dataset_id, urlpath, layer_groups
                        )

            f.write(lines)
            f.close()

    @property
    def catalog(self):
        """Write then open the catalog."""

        if not hasattr(self, "_catalog"):

            self.write_catalog()
            # if we already know there aren't any dataset_ids
            # don't try to read catalog
            if not self.search_results == {}:
                catalog = intake.open_catalog(self.catalog_name)
            else:
                catalog = None
            self._catalog = catalog

        return self._catalog

    @property
    def dataset_ids(self):
        """Find dataset_ids for server.

        Notes
        -----
        The dataset_ids are read from the catalog, so the catalog is created
        before this can happen.

        The number of dataset_ids can change if a variable is removed from the
        list of variables and this is rerun.
        """

        if not hasattr(self, "_dataset_ids") or (
            self.variables and (len(self.variables) != self.num_variables)
        ):
            if self.catalog is not None:
                self._dataset_ids = list(self.catalog)
            else:
                self._dataset_ids = []

            # update number of variables
            if self.variables:
                self.num_variables = len(self.variables)

        return self._dataset_ids

    def meta_by_dataset(self, dataset_id):
        """Return the catalog metadata for a single dataset_id.

        TO DO: Should this return intake-style or a row of the metadata dataframe?
        """

        return self.catalog[dataset_id]

    @property
    def meta(self):
        """Rearrange the individual metadata into a dataframe."""

        if not hasattr(self, "_meta"):

            data = []
            for dataset_id in self.dataset_ids:
                meta = self.meta_by_dataset(dataset_id)
                columns = ["download_url"] + list(
                    meta.metadata.keys()
                )  # this only needs to be set once
                data.append([meta.urlpath] + list(meta.metadata.values()))
            if len(self.dataset_ids) > 0:
                self._meta = pd.DataFrame(
                    index=self.dataset_ids, columns=columns, data=data
                )
            else:
                self._meta = None

        return self._meta

    def data_by_dataset(self, dataset_id):
        """Return the data for a single dataset_id.

        Returns
        -------
        A tuple of (dataset_id, data), where data type depends on `self.axds_type`:
        If `self.axds_type=='platform2'`: a pandas DataFrame
        If `self.axds_type=='layer_group'`: an xarray Dataset

        Notes
        -----
        Read behavior depends on `axds_type`:

        * If `self.axds_type=='platform2'`: data is read into memory with dask.
        * If `self.axds_type=='layer_group'`: data is pointed to with dask but
          nothing is read in except metadata associated with the xarray Dataset.
        """

        if self.axds_type == "platform2":

            if self.filetype == "csv":
                # read units from metadata variable meta_url for columns
                variables = pd.read_json(self.meta.loc[dataset_id]["meta_url"])[
                    "variables"
                ]
                # .to_dask().compute() seems faster than read but
                # should do more comparisons
                data = self.catalog[dataset_id].to_dask().compute()
                data = data.set_index("time")
                data = data[self.kw["min_time"] : self.kw["max_time"]]

                units = []
                for col in data.columns:
                    try:
                        units.append(variables.loc[col]["attributes"]["units"])
                    except:
                        units.append("")

                # add units to 2nd header row
                data.columns = pd.MultiIndex.from_tuples(zip(data.columns, units))

            elif self.filetype == "netcdf":
                # this downloads the http-served file to cache I think
                download_url = self.catalog[dataset_id].urlpath
                infile = fsspec.open(f"simplecache::{download_url}")
                data = xr.open_dataset(infile.open())  # , engine='h5netcdf')
                # we need 'time' as a dimension for the subsequent line to work
                dim = [
                    dim for dim, size in data.dims.items() if size == data.cf["T"].size
                ]
                if len(dim) > 0:
                    data = data.swap_dims({dim[0]: data.cf["T"].name})
                # .swap_dims({"profile": "time"})
                # filter by time
                data = data.cf.sel(T=slice(self.kw["min_time"], self.kw["max_time"]))

        elif self.axds_type == "layer_group":
            if self.catalog[dataset_id].urlpath is not None:
                try:
                    data = self.catalog[dataset_id].to_dask()

                    # preprocess to avoid a sometimes-problem:
                    # try to fix key error assuming it is the following problem:
                    # KeyError: "cannot represent labeled-based slice indexer for dimension 'time' with a slice over integer positions; the index is unsorted or non-unique"
                    try:
                        _, index = np.unique(data.cf["T"], return_index=True)
                        data = data.cf.isel(T=index)

                        # filter by time
                        data = data.cf.sel(
                            T=slice(self.kw["min_time"], self.kw["max_time"])
                        )
                    except KeyError as e:
                        logger.exception(e)
                        logger.warning("Could not subset in time.")
                        pass

                except Exception as e:
                    logger.exception(e)
                    logger.warning(
                        f"data was not read in for dataset_id {dataset_id} with url path {self.catalog[dataset_id].urlpath} and description {self.catalog[dataset_id].description}."
                    )
                    data = None

        # return (dataset_id, data)
        return data

    # @property
    def data(self, dataset_ids=None):
        """Read in data for some or all dataset_ids.

        NOT USED CURRENTLY

        Once data is read in for a dataset_ids, it is remembered.

        See full documentation in `utils.load_data()`.
        """

        output = odg.utils.load_data(self, dataset_ids)
        return output

    def save(self):
        """Save datasets locally."""

        for dataset_id, data in self.data().items():
            # dataframe
            if self.data_type == "csv":
                filename = (
                    f'{dataset_id}_{self.kw["min_time"]}_{self.kw["max_time"]}.csv.gz'
                )
                path_file = odg.path_files.joinpath(filename)
                data.to_csv(path_file)

            # dataset
            elif self.data_type == "nc":
                filename = (
                    f'{dataset_id}_{self.kw["min_time"]}_{self.kw["max_time"]}.nc'
                )
                path_file = odg.path_files.joinpath(filename)
                data.to_netcdf(path_file)


class region(AxdsReader):
    """Inherits from AxdsReader to search over a region of space and time.

    Attributes
    ----------
    kw: dict
      Contains space and time search constraints: `min_lon`, `max_lon`,
      `min_lat`, `max_lat`, `min_time`, `max_time`.
    variables: string or list
      Variable names if you want to limit the search to those. There is
      different behavior depending on `axds_type`:

      * 'platform2': the variable name or names must be from the list available in `odg.all_variables('axds')` and pass the check in `odg.check_variables('axds', variables)`.
      * 'layer_group': the variable name or names will be searched for as a query so just do your best with the names and experiment.

      Alternatively, if the user inputs criteria, variables can be a
      list of the keys from criteria.
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
            Can contain arguments to pass onto the base AxdsReader class
            (catalog_name, parallel, axds_type). The dict entries to initialize
            this class are:

            * kw: dict
              Contains space and time search constraints: `min_lon`, `max_lon`, `min_lat`, `max_lat`, `min_time`, `max_time`.
            * variables: string or list, optional
              Variable names if you want to limit the search to those. There is
              different behavior depending on `axds_type`:

              * 'platform2': the variable name or names must be from the list available in `odg.all_variables('axds')` and pass the check in
                `odg.check_variables('axds', variables)`.
              * 'layer_group': the variable name or names will be searched for
                as a query so just do your best with the names and experiment.

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
        ax_kwargs = {
            "catalog_name": kwargs.get("catalog_name", None),
            "parallel": kwargs.get("parallel", True),
            "axds_type": kwargs.get("axds_type", "platform2"),
        }
        AxdsReader.__init__(self, **ax_kwargs)

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

        # make sure variables are on parameter list if platform2
        if (variables is not None) and (self.axds_type == "platform2"):
            # User is using criteria and variable nickname approach
            if self.criteria and all(var in self.criteria for var in variables):
                variables = odg.select_variables("axds", self.criteria, variables)

            # user is inputting specific reader variable names
            else:
                odg.check_variables("axds", variables)
            # record the number of variables so that a user can change it and
            # the change can be compared.
            self.num_variables = len(variables)
        else:
            self.num_variables = 0

        self.variables = variables


class stations(AxdsReader):
    """Inherits from AxdsReader to search for 1+ stations or dataset_ids.

    Attributes
    ----------
    kw: dict
        Contains time search constraints: `min_time`, `max_time`.
        If not input, all time will be used.
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
            Can contain arguments to pass onto the base AxdsReader class
            (catalog_name, parallel, axds_type). The dict entries to initialize
            this class are:
            * kw: dict, optional
              Contains time search constraints: `min_time`, `max_time`.
              If not input, all time will be used.
            * stations: string, list, optional
              Input station names as they might be commonly known and therefore
              can be searched for as a query term. The station names can be
              input as something like "TABS B" or "8771972" and has pretty good
              success.
              Or, use this option if you know the exact dataset_ids for
              the data you want and `axds_type=='platform2'`. These need to be
              the dataset_ids corresponding to the databases that are being
              searched, so in this case they need to be the Axiom packrat
              uuid's.
              If `axds_type=='layer_group'`, input the layer_group uuids you
              want to search for.

        Notes
        -----
        The axds_type needs to match the station name or dataset_id you are
        searching for.
        """
        assert isinstance(kwargs, dict), "input arguments as dictionary"
        ax_kwargs = {
            "catalog_name": kwargs.get("catalog_name", None),
            "parallel": kwargs.get("parallel", True),
            "axds_type": kwargs.get("axds_type", "platform2"),
        }
        # this inherits AxdsReader's attributes and functions into self
        AxdsReader.__init__(self, **ax_kwargs)

        kw = kwargs.get("kw", None)
        stations = kwargs.get("stations", [])

        self.approach = "stations"

        # I think this isn't true anymore.
        # if self.axds_type == "layer_group":
        #     assertion = 'Input "layer_group" (not module) uuids as station names, not dataset_ids.'
        #     assert dataset_ids is None, assertion

        if not stations == []:
            if not isinstance(stations, list):
                stations = [stations]
        self._stations = stations
        self.variables = None

        # CHECK FOR KW VALUES AS TIMES
        if kw is None:
            kw = {"min_time": "1900-01-01", "max_time": "2100-12-31"}

        self.kw = kw

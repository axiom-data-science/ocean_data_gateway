import hashlib
import logging
import multiprocessing
import os
import re

import intake
import numpy as np
import pandas as pd
import requests
import shapely.wkt

from joblib import Parallel, delayed

import ocean_data_gateway as odg


logger = logging.getLogger(__name__)

# this can be queried with
# search.AxdsReader.reader
reader = "axds"


class AxdsReader:
    def __init__(self, parallel=True, catalog_name=None, axds_type="platform2"):

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

    def url_query(self, query):
        return f"&query={query}"

    def url_variable(self, variable):
        """by parameter group"""
        return f"&tag=Parameter+Group:{variable}"

    def url_region(self):
        # add lonlat filtering
        url_add_box = (
            f'&geom={{"type":"Polygon","coordinates":[[[{self.kw["min_lon"]},{self.kw["min_lat"]}],'
            + f'[{self.kw["max_lon"]},{self.kw["min_lat"]}],'
            + f'[{self.kw["max_lon"]},{self.kw["max_lat"]}],'
            + f'[{self.kw["min_lon"]},{self.kw["max_lat"]}],'
            + f'[{self.kw["min_lon"]},{self.kw["min_lat"]}]]]}}'
        )
        return f"{url_add_box}"

    def url_time(self):
        # add time filtering
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
        """make a list of urls for stations mode."""
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
                # if input stations instead of dataset_ids, using different urls here
                # if self._stations is not None:
                if len(self._stations) > 0:
                    for station in self._stations:
                        urls.append(self.url_builder(self.url_axds_type, query=station))
                else:
                    for dataset_id in self._dataset_ids:
                        urls.append(
                            self.url_builder(self.url_docs_base, dataset_id=dataset_id)
                        )

            self._urls = urls

        return self._urls

    @property
    def search_results(self):

        if not hasattr(self, "_search_results"):

            # loop over urls in case we have stations
            search_results = []
            for url in self.urls:
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
                    # switch to module search results instead of layer_group results
                    module_uuid = search_result["data"]["module_uuid"]
                    # don't repeat unnecessarily
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

        lines = f"""
  {dataset_id}:
    description: {label}
    driver: opendap
    args:
      urlpath: {urlpath}
      engine: 'netcdf4'
      xarray_kwargs:
    metadata:
      variables: {list(layer_groups.values())}
      layer_group_uuids: {list(layer_groups.keys())}
      model_slug: {model_slug}
      geospatial_lon_min: {geospatial_lon_min}
      geospatial_lat_min: {geospatial_lat_min}
      geospatial_lon_max: {geospatial_lon_max}
      geospatial_lat_max: {geospatial_lat_max}
      time_coverage_start: {dataset['start_date_time']}
      time_coverage_end: {dataset['end_date_time']}

"""
        return lines

    def write_catalog(self):

        # if the catalog already exists, don't do this
        if os.path.exists(self.catalog_name):
            return

        else:

            f = open(self.catalog_name, "w")

            if self.axds_type == "platform2":
                lines = "sources:\n"
                for dataset_id, dataset in self.search_results.items():
                    label = dataset["label"].replace(":", "-")
                    urlpath = dataset["source"]["files"]["data.csv.gz"]["url"]
                    metavars = dataset["source"]["meta"]["variables"]
                    Vars, standard_names = zip(
                        *[
                            (key, metavars[key]["attributes"]["standard_name"])
                            for key in metavars.keys()
                            if ("attributes" in metavars[key].keys())
                            and ("standard_name" in metavars[key]["attributes"])
                        ]
                    )
                    P = shapely.wkt.loads(dataset["data"]["geospatial_bounds"])
                    (
                        geospatial_lon_min,
                        geospatial_lat_min,
                        geospatial_lon_max,
                        geospatial_lat_max,
                    ) = P.bounds

                    lines += f"""
  {dataset["uuid"]}:
    description: {label}
    driver: csv
    args:
      urlpath: {urlpath}
      csv_kwargs:
        parse_dates: ['time']
    metadata:
      variables: {Vars}
      standard_names: {standard_names}
      platform_category: {dataset['data']['platform_category']}
      geospatial_lon_min: {geospatial_lon_min}
      geospatial_lat_min: {geospatial_lat_min}
      geospatial_lon_max: {geospatial_lon_max}
      geospatial_lat_max: {geospatial_lat_max}
      id: {dataset["data"]["packrat_source_id"]}
      time_coverage_start: {dataset['start_date_time']}
      time_coverage_end: {dataset['end_date_time']}

"""

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
                            urlpaths.append(
                                search_results_lg["source"]["layers"][0][
                                    "thredds_opendap_url"
                                ][:-5]
                            )
                        else:
                            urlpaths.append("")
                            logger.warning(
                                f"no opendap url for module: module uuid {dataset_id}, layer_group uuid {layer_group_uuid}"
                            )
                            # DO NOT STORE ITEM IN CATALOG IF NOT OPENDAP ACCESSIBLE
                            continue

                    # there may be different urls for different layer_groups
                    # in which case associate the layer_group uuid with the dataset
                    # since the module uuid wouldn't be unique
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

        if not hasattr(self, "_catalog"):

            self.write_catalog()
            # if we already know there aren't any dataset_ids
            # don't try to read catalog
            if not self.search_results == {}:
                #             if (not self.dataset_ids == []) or (not self.search_results == {}):
                catalog = intake.open_catalog(self.catalog_name)
            else:
                catalog = None
            self._catalog = catalog

        return self._catalog

    @property
    def dataset_ids(self):
        """Find dataset_ids for server."""

        if not hasattr(self, "_dataset_ids"):
            if self.catalog is not None:
                self._dataset_ids = list(self.catalog)
            else:
                self._dataset_ids = []

        return self._dataset_ids

    def meta_by_dataset(self, dataset_id):
        """Should this return intake-style or a row of the metadata dataframe?"""

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

        if self.axds_type == "platform2":

            # .to_dask().compute() seems faster than read but
            # should do more comparisons
            data = self.catalog[dataset_id].to_dask().compute()
            data = data.set_index("time")
            data = data[self.kw["min_time"] : self.kw["max_time"]]

        elif self.axds_type == "layer_group":

            if self.catalog[dataset_id].urlpath is not None:
                try:
                    data = self.catalog[dataset_id].to_dask()
                    try:
                        timekey = [
                            coord
                            for coord in data.coords
                            if ("standard_name" in data[coord].attrs)
                            and (data[coord].attrs["standard_name"] == "time")
                        ]
                        assert len(timekey) > 0
                    except:
                        timekey = [
                            coord
                            for coord in data.coords
                            if ("time" in coord) or (coord == "t")
                        ]
                        assert len(timekey) > 0
                    timekey = timekey[0]
                    slicedict = {
                        timekey: slice(self.kw["min_time"], self.kw["max_time"])
                    }
                    data = data.sel(slicedict)
                except KeyError as e:
                    #                     logger.exception(e)
                    #                     logger.warning(f'data was not read in for dataset_id {dataset_id} with url path {self.catalog[dataset_id].urlpath} and description {self.catalog[dataset_id].description}.')

                    # try to fix key error assuming it is the following problem:
                    # KeyError: "cannot represent labeled-based slice indexer for dimension 'time' with a slice over integer positions; the index is unsorted or non-unique"
                    try:
                        timekey = [
                            coord
                            for coord in data.coords
                            if ("standard_name" in data[coord].attrs)
                            and (data[coord].attrs["standard_name"] == "time")
                        ]
                        assert len(timekey) > 0
                    except:
                        timekey = [
                            coord
                            for coord in data.coords
                            if ("time" in coord) or (coord == "t")
                        ]
                        assert len(timekey) > 0
                    timekey = timekey[0]

                    slicedict = {
                        timekey: slice(self.kw["min_time"], self.kw["max_time"])
                    }
                    _, index = np.unique(data[timekey], return_index=True)
                    data = data.isel({timekey: index}).sel(slicedict)
                except Exception as e:
                    logger.exception(e)
                    logger.warning(
                        f"data was not read in for dataset_id {dataset_id} with url path {self.catalog[dataset_id].urlpath} and description {self.catalog[dataset_id].description}."
                    )
                    data = None
            else:
                data = None

        return (dataset_id, data)

    #         return (dataset_id, self.catalog[dataset_id].read())

    # @property
    def data(self):
        """Do I need to worry about intake caching?"""

        if not hasattr(self, "_data"):

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

            self._data = dds

        return self._data

    def save(self):
        """save data locally """

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

    def all_variables(self):
        """Not relevant for layer_group"""

        path_fname = odg.variables_path.joinpath("parameter_group_names.txt")
        path_csv_fname = odg.variables_path.joinpath("axds_platform2_variable_list.csv")
        # read in Axiom Search parameter group names
        # save to file
        if path_csv_fname.is_file():
            df = pd.read_csv(path_csv_fname, index_col="variable")
        else:
            print(
                "Please wait while the list of available variables is made. This only happens once."
            )
            os.system(
                f'curl -sSL -H "Accept: application/json" "https://search.axds.co/v2/search" | jq -r \'.tags["Parameter Group"][] | "\(.label) \(.count)"\' > {path_fname}'
            )

            # read in parameter group names
            f = open(path_fname, "r")
            parameters_temp = f.readlines()
            f.close()
            #         parameters = [parameter.strip('\n') for parameter in parameters]
            parameters = {}
            for parameter in parameters_temp:
                parts = parameter.strip("\n").split()
                name = " ".join(parts[:-1])
                count = parts[-1]
                parameters[name] = count

            df = pd.DataFrame()
            df["variable"] = parameters.keys()
            df["count"] = parameters.values()
            df = df.set_index("variable")
            df.to_csv(path_csv_fname)

        return df

    def search_variables(self, variables):
        """Find valid variables names to use.

        Call with `search_variables()` to return the list of possible names.
        Call with `search_variables('salinity')` to return relevant names.
        """

        if not isinstance(variables, list):
            variables = [variables]

        # set up search for input variables
        search = f"(?i)"
        for variable in variables:
            search += f".*{variable}|"
        search = search.strip("|")

        r = re.compile(search)

        df = self.all_variables()
        parameters = df.index

        matches = list(filter(r.match, parameters))

        # return parameters that match input variable strings
        return df.loc[matches].sort_values("count", ascending=False)

    def check_variables(self, variables, verbose=False):

        assertion = f'Variables are only used to filter the search for \
                    \n`axds_type="platform2". Currently, \
                    \naxds_type={self.axds_type}.'
        assert self.axds_type == "platform2", assertion

        if not isinstance(variables, list):
            variables = [variables]

        parameters = list(self.all_variables().index)

        # for a variable to exactly match a parameter
        # this should equal 1
        count = []
        for variable in variables:
            count += [parameters.count(variable)]

        condition = np.allclose(count, 1)

        assertion = f"The input variables are not exact matches to parameter groups. \
                     \nCheck all parameter group values with `AxdsReader().all_variables()` \
                     \nor search parameter group values with `AxdsReader().search_variables({variables})`.\
                     \n\n Try some of the following variables:\n{str(self.search_variables(variables))}"

        assert condition, assertion

        if condition and verbose:
            print("all variables are matches!")

    # Search for stations by region


class region(AxdsReader):
    #     def region(self, kw, axds_type='platform2', variables=None):
    #         '''HOW TO INCORPORATE VARIABLE NAMES?'''

    def __init__(self, kwargs):
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

        #         # can be 'platform2' or 'layer_group'
        #         assert axds_type in ['platform2','layer_group'], 'variable `axds_type` must be "platform2" or "layer_group"'
        #         self.axds_type = axds_type

        #         self.url_axds_type = f'{self.url_search_base}&type={self.axds_type}'

        # run checks for KW
        # check for lon/lat values and time
        self.kw = kw

        # #         self.data_type = data_type
        if (variables is not None) and (not isinstance(variables, list)):
            variables = [variables]

        # make sure variables are on parameter list
        if variables is not None:
            self.check_variables(variables)

        self.variables = variables


#         # DOESN'T CURRENTLY LIMIT WHICH VARIABLES WILL BE FOUND ON EACH SERVER

#         return self


class stations(AxdsReader):
    #     def stations(self, dataset_ids=None, stations=None, kw=None, axds_type='platform2'):
    #         '''

    #         Use keyword dataset_ids if you already know the database-
    #         specific ids. Otherwise, use the keyword stations and the
    #         database-specific ids will be searched for. The station
    #         ids can be input as something like "TABS B" and will be
    #         searched for as "TABS AND B" and has pretty good success.

    #         Treat dataset_ids and stations the same since either way we
    #         need to search for them. Have both to match the erddap
    #         syntax.

    #         WHAT ABOUT VARIABLES?
    #         '''

    def __init__(self, kwargs):
        assert isinstance(kwargs, dict), "input arguments as dictionary"
        ax_kwargs = {
            "catalog_name": kwargs.get("catalog_name", None),
            "parallel": kwargs.get("parallel", True),
            "axds_type": kwargs.get("axds_type", "platform2"),
        }
        AxdsReader.__init__(self, **ax_kwargs)

        kw = kwargs.get("kw", None)
        dataset_ids = kwargs.get("dataset_ids", None)
        stations = kwargs.get("stations", [])

        self.approach = "stations"

        #         self.catalog_name = os.path.join('..','catalogs',f'catalog_stations_{pd.Timestamp.now().isoformat()[:19]}.yml')

        #         # we want all the data associated with stations
        #         self.standard_names = None

        # UPDATE SINCE NOW THERE IS A DIFFERENCE BETWEEN STATION AND DATASET
        if dataset_ids is not None:
            if not isinstance(dataset_ids, list):
                dataset_ids = [dataset_ids]
            #             self._stations = dataset_ids
            self._dataset_ids = dataset_ids

        #         assert (if dataset_ids is not None)
        # assert that dataset_ids can't be something if axds_type is layer_group
        # use stations instead, and don't use module uuid, use layer_group uuid

        if not stations == []:
            if not isinstance(stations, list):
                stations = [stations]
        self._stations = stations

        #         self.dataset_ids

        self.variables = None

        # CHECK FOR KW VALUES AS TIMES
        if kw is None:
            kw = {"min_time": "1900-01-01", "max_time": "2100-12-31"}

        self.kw = kw

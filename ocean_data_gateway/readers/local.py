"""
Reader for local files.
"""

import hashlib
import logging
import multiprocessing
import os

import intake
import pandas as pd

from joblib import Parallel, delayed

import ocean_data_gateway as odg


logger = logging.getLogger(__name__)

# this can be queried with
# search.LocalReader.reader
reader = "local"


class LocalReader:
    """
    This class searches local files.

    Attributes
    ----------
    parallel: boolean
        If True, run with simple parallelization using `multiprocessing`.
        If False, run serially.
    catalog_name: string
        Input catalog path if you want to use an existing catalog.
    filenames: string, list
        Specific file locations from which to read data.
    kw: dict, optional
      Contains space and time search constraints: `min_lon`, `max_lon`,
      `min_lat`, `max_lat`, `min_time`, `max_time`.
    name: string
        f'axds_{axds_type}' so 'axds_platform2' or 'axds_layer_group'
    reader: string
        Reader name: AxdsReader

    TO DO: Can this reader be used for remote datasets but for
    which we know the specific file location?
    """

    def __init__(self, parallel=True, catalog_name=None, filenames=None, kw=None):
        """
        Parameters
        ----------
        parallel: boolean, optional
            If True, run with simple parallelization using `multiprocessing`.
            If False, run serially.
        catalog_name: string, optional
            Input catalog path if you want to use an existing catalog.
        filenames: string, list
            Specific file locations from which to read data.
        kw: dict
            Contains space and time search constraints: `min_lon`, `max_lon`,
            `min_lat`, `max_lat`, `min_time`, `max_time`.

        Notes
        -----
        All input data is currently used, regardless of whether `kw` is input
        with constraints on lon, lat, or time.

        There is no real difference between searching with `region` or
        `stations` for this reader.
        """

        self.parallel = parallel

        if catalog_name is None:
            name = f"{pd.Timestamp.now().isoformat()}"
            hash_name = hashlib.sha256(name.encode()).hexdigest()[:7]
            catalog_path = odg.catalogs_path.joinpath(f"catalog_{hash_name}.yml")
            self.catalog_name = catalog_path
        else:
            self.catalog_name = catalog_name
            # if catalog_name already exists, read it in to save time
            self.catalog

        if (filenames is not None) and (not isinstance(filenames, list)):
            filenames = [filenames]
        self.filenames = filenames

        if kw is None:
            kw = {"min_time": "1900-01-01", "max_time": "2100-12-31"}

        self.kw = kw

        if (filenames is None) and (catalog_name is None):
            self._dataset_ids = []
            logger.warning(
                f"no datasets for LocalReader with catalog_name {catalog_name} and filenames {filenames}."
            )

        # name
        self.name = "local"

        self.reader = "LocalReader"

    def write_catalog(self):
        """Write catalog file."""

        # if the catalog already exists, don't do this
        if os.path.exists(self.catalog_name):
            return

        else:
            lines = "sources:\n"

            for filename in self.filenames:

                if "csv" in filename:
                    file_intake = intake.open_csv(filename)
                    data = file_intake.read()
                    metadata = {
                        "variables": list(data.columns.values),
                        "geospatial_lon_min": float(data["longitude"].min()),
                        "geospatial_lat_min": float(data["latitude"].min()),
                        "geospatial_lon_max": float(data["longitude"].max()),
                        "geospatial_lat_max": float(data["latitude"].max()),
                        "time_coverage_start": data["time"].min(),
                        "time_coverage_end": data["time"].max(),
                    }
                    file_intake.metadata = metadata

                elif "nc" in filename:
                    file_intake = intake.open_netcdf(filename)
                    data = file_intake.read()
                    coords = list(data.coords.keys())
                    timekey = [
                        coord
                        for coord in coords
                        if ("time" in data[coord].attrs.values())
                        or ("T" in data[coord].attrs.values())
                    ]
                    if len(timekey) > 0:
                        timekey = timekey[0]
                        time_coverage_start = str(data[timekey].min().values)
                        time_coverage_end = str(data[timekey].max().values)
                    else:
                        time_coverage_start = ""
                        time_coverage_end = ""
                    lonkey = [
                        coord
                        for coord in coords
                        if ("lon" in data[coord].attrs.values())
                        or ("X" in data[coord].attrs.values())
                    ]
                    if len(lonkey) > 0:
                        lonkey = lonkey[0]
                        geospatial_lon_min = float(data[lonkey].min())
                        geospatial_lon_max = float(data[lonkey].max())
                    else:
                        geospatial_lon_min = ""
                        geospatial_lon_max = ""
                    latkey = [
                        coord
                        for coord in coords
                        if ("lat" in data[coord].attrs.values())
                        or ("Y" in data[coord].attrs.values())
                    ]
                    if len(latkey) > 0:
                        latkey = latkey[0]
                        geospatial_lat_min = float(data[latkey].min())
                        geospatial_lat_max = float(data[latkey].max())
                    else:
                        geospatial_lat_min = ""
                        geospatial_lat_max = ""
                    metadata = {
                        "coords": coords,
                        "variables": list(data.data_vars.keys()),
                        "time_variable": timekey,
                        "lon_variable": lonkey,
                        "lat_variable": latkey,
                        "geospatial_lon_min": geospatial_lon_min,
                        "geospatial_lon_max": geospatial_lon_max,
                        "geospatial_lat_min": geospatial_lat_min,
                        "geospatial_lat_max": geospatial_lat_max,
                        "time_coverage_start": time_coverage_start,
                        "time_coverage_end": time_coverage_end,
                    }
                    file_intake.metadata = metadata

                file_intake.name = filename.split("/")[-1]
                lines += file_intake.yaml().strip("sources:")

            f = open(self.catalog_name, "w")
            f.write(lines)
            f.close()

    @property
    def catalog(self):
        """Write then open catalog."""

        if not hasattr(self, "_catalog"):

            self.write_catalog()
            catalog = intake.open_catalog(self.catalog_name)
            self._catalog = catalog

        return self._catalog

    @property
    def dataset_ids(self):
        """Find dataset_ids for catalog.

        Notes
        -----
        The dataset_ids are read from the catalog, so the catalog is created
        before this can happen.
        """

        if not hasattr(self, "_dataset_ids"):
            self._dataset_ids = list(self.catalog)

        return self._dataset_ids

    def meta_by_dataset(self, dataset_id):
        """Return the catalog metadata for a single dataset_id.

        TODO: Should this return intake-style or a row of the metadata dataframe?
        """

        return self.catalog[dataset_id]

    @property
    def meta(self):
        """Rearrange the individual metadata into a dataframe."""

        if not hasattr(self, "_meta"):

            if self.dataset_ids == []:
                self._meta = None
            else:
                # set up columns which might be different for datasets
                columns = ["download_url"]
                for dataset_id in self.dataset_ids:
                    meta = self.meta_by_dataset(dataset_id)
                    columns += list(meta.metadata.keys())
                columns = set(columns)  # take unique column names

                self._meta = pd.DataFrame(index=self.dataset_ids, columns=columns)
                for dataset_id in self.dataset_ids:
                    meta = self.meta_by_dataset(dataset_id)
                    self._meta.loc[dataset_id]["download_url"] = meta.urlpath
                    self._meta.loc[dataset_id, list(meta.metadata.keys())] = list(
                        meta.metadata.values()
                    )
                    # self._meta.loc[dataset_id][meta.metadata.keys()] = meta.metadata.values()
                    # data.append([meta.urlpath] + list(meta.metadata.values()))
                # self._meta = pd.DataFrame(index=self.dataset_ids, columns=columns, data=data)

        return self._meta

    def data_by_dataset(self, dataset_id):
        """Return the data for a single dataset_id.

        Returns
        -------
        A tuple of (dataset_id, data), where data type is a pandas DataFrame.

        Notes
        -----
        Data is read into memory.

        TODO: SHOULD I INCLUDE TIME RANGE?
        """

        data = self.catalog[dataset_id].read()
        #         data = data.set_index('time')
        #         data = data[self.kw['min_time']:self.kw['max_time']]

        return (dataset_id, data)

    #         return (dataset_id, self.catalog[dataset_id].read())

    # @property
    def data(self):
        """Read in data for all dataset_ids.

        Returns
        -------
        A dictionary with keys of the dataset_ids and values the data of type:
        If `filename` is a csv file: a pandas DataFrame
        If `filename` is a netcdf file: an xarray Dataset

        Notes
        -----
        This is either done in parallel with the `multiprocessing` library or
        in serial.
        """

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


class region(LocalReader):
    """Inherits from LocalReader to search over a region of space and time.

    Attributes
    ----------
    kw: dict
      Contains space and time search constraints: `min_lon`, `max_lon`,
      `min_lat`, `max_lat`, `min_time`, `max_time`.
    variables: string or list
      Variable names if you want to limit the search to those. This is currently
      ignored.
    approach: string
        approach is defined as 'region' for this class.
    """

    def __init__(self, kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Can contain arguments to pass onto the base AxdsReader class
            (catalog_name, parallel, filenames). The dict entries to initialize
            this class are:

            * kw: dict
              Contains space and time search constraints: `min_lon`, `max_lon`,
              `min_lat`, `max_lat`, `min_time`, `max_time`. Not used to filter
              data currently.
            * variables: string or list, optional
              Variable names if you want to limit the search to those. This is
              not used to filter data currently.
        """
        assert isinstance(kwargs, dict), "input arguments as dictionary"
        lo_kwargs = {
            "catalog_name": kwargs.get("catalog_name", None),
            "filenames": kwargs.get("filenames", None),
            "parallel": kwargs.get("parallel", True),
        }
        LocalReader.__init__(self, **lo_kwargs)

        kw = kwargs["kw"]
        variables = kwargs.get("variables", None)

        self.approach = "region"

        self._stations = None

        # run checks for KW
        # check for lon/lat values and time
        self.kw = kw

        if (variables is not None) and (not isinstance(variables, list)):
            variables = [variables]

        self.variables = variables


class stations(LocalReader):
    """Inherits from LocalReader to search for 1+ stations or dataset_ids.

    Attributes
    ----------
    kw: dict
      Contains space and time search constraints: `min_lon`, `max_lon`,
      `min_lat`, `max_lat`, `min_time`, `max_time`.
    approach: string
        approach is defined as 'stations' for this class.
    """

    def __init__(self, kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Can contain arguments to pass onto the base LocalReader class
            (catalog_name, parallel, filenames). The dict entries to initialize
            this class are:
            * kw: dict, optional
              Contains space and time search constraints: `min_lon`, `max_lon`,
              `min_lat`, `max_lat`, `min_time`, `max_time`.
        """
        assert isinstance(kwargs, dict), "input arguments as dictionary"
        loc_kwargs = {
            "catalog_name": kwargs.get("catalog_name", None),
            "filenames": kwargs.get("filenames", None),
            "parallel": kwargs.get("parallel", True),
        }
        LocalReader.__init__(self, **loc_kwargs)

        kw = kwargs.get("kw", None)
        self.approach = "stations"

        # CHECK FOR KW VALUES AS TIMES
        if kw is None:
            kw = {"min_time": "1900-01-01", "max_time": "2100-12-31"}

        self.kw = kw

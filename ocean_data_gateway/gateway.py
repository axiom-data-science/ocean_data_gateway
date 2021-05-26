"""
This controls and connects to the individual readers.
"""

import ocean_data_gateway as odg


# MAYBE SHOULD BE ABLE TO INITIALIZE THE CLASS WITH ONLY METADATA OR DATASET NAMES?
# to skip looking for the datasets


class Gateway(object):
    """
    Wraps together the individual readers in order to have a single way to
    search.

    Attributes
    ----------
    kwargs_all: dict
        Input keyword arguments that are not specific to one of the readers.
        These may include "approach", "parallel", "kw" containing the time and
        space region to search for, etc.
    kwargs: dict
        Keyword arguments that contain specific arguments for the readers.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kw: dict
          Contains space and time search constraints: `min_lon`, `max_lon`,
          `min_lat`, `max_lat`, `min_time`, `max_time`.
        approach: string
            approach is defined as 'stations' or 'region' depending on user
            choice.
        parallel: boolean, optional
            If True, run with simple parallelization using `multiprocessing`.
            If False, run serially. True by default. If input in this manner,
            the same value is used for all readers. If input by individual
            reader dictionary, the value can vary by reader.
        readers: ocean_data_gateway Reader, list of readers, optional
            Use this to use fewer than the full set of readers. For example,
            `readers=odg.erddap` or to specifically include all by name
            `readers = [odg.ErddapReader, odg.axdsReader, odg.localReader]`.
        erddap: dict, optional
            Dictionary of reader specifications. For example,
            `erddap={'known_server': 'ioos'}`. See odg.erddap.ErddapReader for
            more input options.
        axds: dict, optional
            Dictionary of reader specifications. For example,
            `axds={'axds_type': 'platform2'}`. See odg.axds.AxdsReader for
            more input options.
        local: dict, optional
            Dictionary of reader specifications. For example,
            `local={'filenames': filenames}` for a list of filenames.
            See odg.local.LocalReader for more input options.

        Notes
        -----
        To select search variables, input the variable names to each reader
        individually in the format `erddap={'variables': [list of variables]}`.
        Make sure that the variable names are correct for each individual
        reader. Check individual reader docs for more information.

        Input keyword arguments that are not specific to one of the readers will be collected in local dictionary kwargs_all. These may include "approach", "parallel", "kw" containing the time and space region to search for, etc.

        Input keyword arguments that are specific to readers will be collected
        in local dictionary kwargs.
        """

        # make sure only known keys are input in kwargs
        unknown_keys = set(list(kwargs.keys())) - set(odg.keys_kwargs)
        assertion = f"keys into Gateway {unknown_keys} are unknown."
        assert len(unknown_keys) == 0, assertion

        # set up a dictionary for general input kwargs
        exclude_keys = ["erddap", "axds", "local"]
        kwargs_all = {
            k: kwargs[k] for k in set(list(kwargs.keys())) - set(exclude_keys)
        }

        self.kwargs_all = kwargs_all

        # default approach is region
        if "approach" not in self.kwargs_all:
            self.kwargs_all["approach"] = "region"

        assertion = '`approach` has to be "region" or "stations"'
        assert self.kwargs_all["approach"] in ["region", "stations"], assertion

        # if "kw" not in self.kwargs_all:
        #     kw = {
        #         "min_lon": -124.0,
        #         "max_lon": -122.0,
        #         "min_lat": 38.0,
        #         "max_lat": 40.0,
        #         "min_time": "2021-4-1",
        #         "max_time": "2021-4-2",
        #     }
        #     self.kwargs_all["kw"] = kw

        self.kwargs = kwargs
        self.sources

    @property
    def sources(self):
        """Set up data sources (readers).

        Notes
        -----
        All readers are included by default (readers as listed in odg._SOURCES). See
         __init__ for options.
        """

        if not hasattr(self, "_sources"):

            # allow user to override what readers to use
            if "readers" in self.kwargs_all.keys():
                SOURCES = self.kwargs_all["readers"]
                if not isinstance(SOURCES, list):
                    SOURCES = [SOURCES]
            else:
                SOURCES = odg._SOURCES

            # loop over data sources to set them up
            sources = []
            for source in SOURCES:
                #                 print(source.reader)

                # in case of important options for readers
                # but the built in options are ignored for a reader
                # if one is input in kwargs[source.reader]
                if source.reader in odg.OPTIONS.keys():
                    reader_options = odg.OPTIONS[source.reader]
                    reader_key = list(reader_options.keys())[0]
                    # if the user input their own option for this, use it instead
                    # this makes it loop once
                    if (source.reader in self.kwargs.keys()) and (
                        reader_key in self.kwargs[source.reader]
                    ):
                        #                         reader_values = [None]
                        reader_values = self.kwargs[source.reader][reader_key]
                    else:
                        reader_values = list(reader_options.values())[0]
                else:
                    reader_key = None
                    # this is to make it loop once for cases without
                    # extra options like localReader
                    reader_values = [None]
                if not isinstance(reader_values, list):
                    reader_values = [reader_values]

                # catch if the user is putting in a set of variables to match
                # the set of reader options
                if (source.reader in self.kwargs) and (
                    "variables" in self.kwargs[source.reader]
                ):
                    variables_values = self.kwargs[source.reader]["variables"]
                    if not isinstance(variables_values, list):
                        variables_values = [variables_values]
                #                     if len(reader_values) == variables_values:
                #                         variables_values
                else:
                    variables_values = [None] * len(reader_values)

                # catch if the user is putting in a set of dataset_ids to match
                # the set of reader options
                if (source.reader in self.kwargs) and (
                    "dataset_ids" in self.kwargs[source.reader]
                ):
                    dataset_ids_values = self.kwargs[source.reader]["dataset_ids"]
                    if not isinstance(dataset_ids_values, list):
                        dataset_ids_values = [dataset_ids_values]
                #                     if len(reader_values) == variables_values:
                #                         variables_values
                else:
                    dataset_ids_values = [None] * len(reader_values)

                for option, variables, dataset_ids in zip(
                    reader_values, variables_values, dataset_ids_values
                ):
                    # setup reader with kwargs for that reader
                    # prioritize input kwargs over default args
                    # NEED TO INCLUDE kwargs that go to all the readers
                    args = {}
                    args_in = {
                        **args,
                        **self.kwargs_all,
                        #                                reader_key: option,
                        #                                **self.kwargs[source.reader],
                    }

                    if source.reader in self.kwargs.keys():
                        args_in = {
                            **args_in,
                            **self.kwargs[source.reader],
                        }

                    args_in = {**args_in, reader_key: option}

                    # deal with variables separately
                    args_in = {
                        **args_in,
                        "variables": variables,
                    }

                    # deal with dataset_ids separately
                    args_in = {
                        **args_in,
                        "dataset_ids": dataset_ids,
                    }

                    if self.kwargs_all["approach"] == "region":
                        reader = source.region(args_in)
                    elif self.kwargs_all["approach"] == "stations":
                        reader = source.stations(args_in)

                    sources.append(reader)

            self._sources = sources

        return self._sources

    @property
    def dataset_ids(self):
        """Find dataset_ids for each source/reader.

        Returns
        -------
        A list of dataset_ids where each entry in the list corresponds to one source/reader, which in turn contains a list of dataset_ids.
        """

        if not hasattr(self, "_dataset_ids"):

            dataset_ids = []
            for source in self.sources:

                dataset_ids.append(source.dataset_ids)

            self._dataset_ids = dataset_ids

        return self._dataset_ids

    @property
    def meta(self):
        """Find and return metadata for datasets.

        Returns
        -------
        A list with an entry for each reader. Each entry in the list contains a pandas DataFrames of metadata for that reader.

        Notes
        -----
        This is done by querying each data source function for metadata and
        then using the metadata for quick returns.

        This will not rerun if the metadata has already been found.

        Different sources have different metadata, though certain attributes
        are always available.

        TO DO: SEPARATE DATASOURCE FUNCTIONS INTO A PART THAT RETRIEVES THE
        DATASET_IDS AND METADATA AND A PART THAT READS IN THE DATA.
        """

        if not hasattr(self, "_meta"):

            # loop over data sources to read in metadata
            meta = []
            for source in self.sources:

                meta.append(source.meta)

            self._meta = meta

        return self._meta

    @property
    def data(self):
        """Return the data, given metadata.

        Notes
        -----
        This is either done in parallel with the `multiprocessing` library or
        in serial.
        """

        if not hasattr(self, "_data"):

            # loop over data sources to read in data
            data = []
            for source in self.sources:

                data.append(source.data)

            self._data = data

        return self._data

"""
This controls and connects to the individual readers.
"""

import cf_xarray  # isort:skip
from cf_xarray.units import units  # isort:skip
import pint_xarray  # isort:skip

pint_xarray.unit_registry = units  # isort:skip

import pandas as pd  # noqa: E402
import pint_xarray  # noqa: E402
import xarray as xr  # noqa: E402

from ioos_qc.config import QcConfig  # noqa: E402

import ocean_data_gateway as odg  # noqa: E402

from ocean_data_gateway import Reader  # noqa: E402


class Gateway(Reader):
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

    def __init__(self, *args, **kwargs):
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

        Notes
        -----
        To select search variables, input the variable names to each reader
        individually in the format `erddap={'variables': [list of variables]}`.
        Make sure that the variable names are correct for each individual
        reader. Check individual reader docs for more information.

        Alternatively, the user can input `criteria` and then input as variables
        the nicknames provided in `criteria` for variable names. These should
        then be input generally, not to an individual reader.

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

        # check for custom criteria to set up cf-xarray
        if "criteria" in self.kwargs_all:
            criteria = self.kwargs_all["criteria"]
            # link to nonlocal dictionary definition
            if isinstance(criteria, str) and criteria[:4] == "http":
                criteria = odg.return_response(criteria)
            cf_xarray.set_options(custom_criteria=criteria)
            self.criteria = criteria
        else:
            self.criteria = None

        # user-input variable definitions for QC
        if "var_def" in self.kwargs_all:
            var_def = self.kwargs_all["var_def"]
            # link to nonlocal dictionary definition
            if isinstance(var_def, str) and var_def[:4] == "http":
                var_def = odg.return_response(var_def)
            self.var_def = var_def
        else:
            self.var_def = None

        # if both criteria and var_def are input by user, make sure the keys
        # in var_def are all available in criteria.
        if self.criteria and self.var_def:
            assertion = (
                "All variable keys in `var_def` must be available in `criteria`."
            )
            assert all(elem in self.criteria for elem in self.var_def), assertion

        self.kwargs = kwargs
        self.sources

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
                # catch scenario where variables input to all readers at once
                elif "variables" in self.kwargs_all:
                    variables_values = [self.kwargs_all["variables"]] * len(
                        reader_values
                    )
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

                    # # deal with dataset_ids separately
                    # args_in = {
                    #     **args_in,
                    #     "dataset_ids": dataset_ids,
                    # }

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

        dataset_ids = []
        for source in self.sources:

            dataset_ids.extend(source.dataset_ids)

        return dataset_ids

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
        """

        if not hasattr(self, "_meta"):

            # loop over data sources to read in metadata
            meta = []
            for source in self.sources:
                meta.append(source.meta)

            # self._meta = meta
            # merge metadata into one DataFrame
            self._meta = pd.concat(meta, axis=1)

        return self._meta

    def data_by_dataset(self, dataset_id):
        """Return the data for a single dataset_id.

        All available sources are checked (in order) for the dataset. Once a
        dataset matching dataset_id is found, it is returned.

        Returns
        -------
        An xarray Dataset

        Notes
        -----
        Data is read into memory.
        """

        for source in self.sources:
            if dataset_id in source.dataset_ids:
                found_data = source[dataset_id]
                return found_data

    @property
    def data(self, dataset_ids=dataset_ids):
        """Return the data, given metadata.

        THIS IS NOW OUTDATED.

        Notes
        -----
        This is either done in parallel with the `multiprocessing` library or
        in serial.
        """

        if not hasattr(self, "_data"):

            # loop over data sources to read in data
            data = []
            for source in self.sources:

                # import pdb; pdb.set_trace()
                data.append(source.data)
                # data.append(source[dataset_ids])
                # data.append(source.data(dataset_ids=dataset_ids))

            # import pdb; pdb.set_trace()
            # # make dict from individual dicts
            # from collections import ChainMap
            #
            # data = ChainMap(*[d() for d in data])
            self._data = data

        return self._data

    def qc(self, dataset_ids=None, verbose=False):
        """Light quality check on data.

        This runs one IOOS QARTOD on data as a first order quality check.
        Only returns data that is quality checked.

        Requires pint for unit handling. Requires user-input `criteria` and
        `var_def` to run.

        This is slow if your data is both chunks of time and space, so this
        should first narrow by both as much as possible.

        Parameters
        ----------
        dataset_ids: str, list, optional
            Read in data for dataset_ids specifically. If none are
            provided, data will be read in for all `self.keys()`.
        verbose: boolean, optional
            If True, report summary statistics on QC flag distribution in datasets.

        Returns
        -------
        Dataset with added variables for each variable in dataset that was checked, with name of [variable]+'_qc'.

        Notes
        -----
        Code has been saved for data in DataFrames, but is changing so
        that data will be in Datasets. This way, can use cf-xarray
        functionality for custom variable names and easier to have
        recognizable units for variables with netcdf than csv.
        """

        assertion = (
            "Need to have custom criteria and variable information defined to run QC."
        )
        assert self.criteria and self.var_def, assertion

        if dataset_ids is None:
            data_ids = (
                self.keys()
            )  # Only return already read-in dataset_ids  # self.dataset_ids
        else:
            data_ids = dataset_ids
            if not isinstance(data_ids, list):
                data_ids = [data_ids]

        data_out = {}
        for data_id in data_ids:
            # access the Dataset
            dd = self[data_id]

            # which custom variable names are in dataset
            varnames = [
                (cf_xarray.accessor._get_custom_criteria(dd, var), var)
                for var in self.var_def.keys()
                if len(cf_xarray.accessor._get_custom_criteria(dd, var)) > 0
            ]
            assert len(varnames) > 0, "no custom names matched in Dataset."
            # dd_varnames are the variable names in the Dataset dd
            # cf_varnames are the custom names we can use to refer to the
            # variables through cf-xarray
            dd_varnames, cf_varnames = zip(*varnames)
            dd_varnames = sum(dd_varnames, [])
            assert len(dd_varnames) == len(
                cf_varnames
            ), "looks like multiple variables might have been identified for a custom variable name"

            # subset to just the boem or requested variables for each df or ds
            if isinstance(dd, pd.DataFrame):
                dd2 = dd[list(varnames.values())]
            elif isinstance(dd, xr.Dataset):
                dd2 = dd.cf[cf_varnames]
                # dd2 = dd[varnames]  # equivalent

            # Preprocess to change salinity units away from 1e-3
            if isinstance(dd, pd.DataFrame):
                # this replaces units in the 2nd column level of 1e-3 with psu
                new_levs = [
                    "psu" if col == "1e-3" else col for col in dd2.columns.levels[1]
                ]
                dd2.columns.set_levels(new_levs, level=1, inplace=True)
            elif isinstance(dd, xr.Dataset):
                for Var in dd2.data_vars:
                    if "units" in dd2[Var].attrs and dd2[Var].attrs["units"] == "1e-3":
                        dd2[Var].attrs["units"] = "psu"
            # run pint quantify on each data structure
            dd2 = dd2.pint.quantify()
            # dd2 = dd2.pint.quantify(level=-1)

            # go through each variable by name to make sure in correct units
            # have to do this in separate loop so that can dequantify afterward
            if isinstance(dd, pd.DataFrame):
                print("NOT IMPLEMENTED FOR DATAFRAME YET")
            elif isinstance(dd, xr.Dataset):
                # form of "temp": "degree_Celsius"
                units_dict = {
                    dd_varname: self.var_def[cf_varname]["units"]
                    for (dd_varname, cf_varname) in zip(dd_varnames, cf_varnames)
                }
                # convert to conventional units
                dd2 = dd2.pint.to(units_dict)

            dd2 = dd2.pint.dequantify()

            # now loop for QARTOD on each variable
            for dd_varname, cf_varname in zip(dd_varnames, cf_varnames):
                # run QARTOD
                qc_config = {
                    "qartod": {
                        "gross_range_test": {
                            "fail_span": self.var_def[cf_varname]["fail_span"],
                            "suspect_span": self.var_def[cf_varname]["suspect_span"],
                        },
                    }
                }
                qc = QcConfig(qc_config)
                qc_results = qc.run(inp=dd2[dd_varname])
                # qc_results = qc.run(inp=dd2.cf[cf_varname])  # this isn't working for some reason

                # put flags into dataset
                new_qc_var = f"{dd_varname}_qc"
                if isinstance(dd, pd.DataFrame):
                    dd2[new_qc_var] = qc_results["qartod"]["gross_range_test"]
                elif isinstance(dd, xr.Dataset):
                    new_data = qc_results["qartod"]["gross_range_test"]
                    dims = dd2[dd_varname].dims
                    dd2[f"{dd_varname}_qc"] = (dims, new_data)

            data_out[data_id] = dd2

        if verbose:
            for dataset_id, dd in data_out.items():
                print(dataset_id)
                qckeys = dd2[[var for var in dd.data_vars if "_qc" in var]]
                for qckey in qckeys:
                    print(qckey)
                    for flag, desc in odg.qcdefs.items():
                        print(
                            f"Flag == {flag} ({desc}): {int((dd[qckey] == int(flag)).sum())}"
                        )

        return data_out

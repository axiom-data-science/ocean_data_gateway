import ocean_data_gateway as odg
# from search.ErddapReader import ErddapReader
# from search.axdsReader import axdsReader
# from search.localReader import localReader

import pandas as pd

# # data functions by data_type
# DATASOURCES_GRID = [hfradar, seaice_extent, seaice_con]
# DATASOURCES_SENSOR = [sensors]
# DATASOURCES_PLATFORM = [sensors, argo]  # has gliders

# GO THROUGH ALL KNOWN SOURCES?
# SOURCES = [ErddapReader(known_server='ioos'),
#            ErddapReader(known_server='coastwatch'),
_SOURCES = [odg.erddap,
           odg.axds,
           odg.local]

OPTIONS = {'erddap': {'known_server': ['ioos', 'coastwatch']},
           'axds': {'axds_type': ['platform2', 'layer_group']}}

# MAYBE SHOULD BE ABLE TO INITIALIZE THE CLASS WITH ONLY METADATA OR DATASET NAMES?
# to skip looking for the datasets

class Gateway(object):

    def __init__(self, **kwargs):# kw=None, variables=None):

        # set up a dictionary for general input kwargs
        exclude_keys = ['erddap', 'axds', 'local']
        kwargs_all = {k: kwargs[k] for k in set(list(kwargs.keys()))
                                          - set(exclude_keys)}

        self.kwargs_all = kwargs_all

        # default approach is region
        if not 'approach' in self.kwargs_all:
            self.kwargs_all['approach'] = 'region'

        assertion = '`approach` has to be "region" or "stations"'
        assert self.kwargs_all['approach'] in ['region','stations'], assertion

        if not 'kw' in self.kwargs_all:
            kw = {
                "min_lon": -124.0,
                "max_lon": -122.0,
                "min_lat": 38.0,
                "max_lat": 40.0,
                "min_time": '2021-4-1',
                "max_time": '2021-4-2',
            }
            self.kwargs_all['kw'] = kw


        self.kwargs = kwargs

        self.sources

    @property
    def sources(self):
        '''Set up data sources.
        '''

        if not hasattr(self, '_sources'):

            # allow user to override what readers to use
            if 'readers' in self.kwargs_all.keys():
                SOURCES = self.kwargs_all['readers']
                if not isinstance(SOURCES, list):
                    SOURCES = [SOURCES]
            else:
                SOURCES = _SOURCES

            # loop over data sources to set them up
            sources = []
            for source in SOURCES:
#                 print(source.reader)

                # in case of important options for readers
                # but the built in options are ignored for a reader
                # if one is input in kwargs[source.reader]
                if (source.reader in OPTIONS.keys()):
                    reader_options = OPTIONS[source.reader]
                    reader_key = list(reader_options.keys())[0]
                    # if the user input their own option for this, use it instead
                    # this makes it loop once
                    if (source.reader in self.kwargs.keys()) and (reader_key in self.kwargs[source.reader]):
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
                if (source.reader in self.kwargs) and ('variables' in self.kwargs[source.reader]):
                    variables_values = self.kwargs[source.reader]['variables']
                    if not isinstance(variables_values, list):
                        variables_values = [variables_values]
#                     if len(reader_values) == variables_values:
#                         variables_values
                else:
                    variables_values = [None]*len(reader_values)

                # catch if the user is putting in a set of dataset_ids to match
                # the set of reader options
                if (source.reader in self.kwargs) and ('dataset_ids' in self.kwargs[source.reader]):
                    dataset_ids_values = self.kwargs[source.reader]['dataset_ids']
                    if not isinstance(dataset_ids_values, list):
                        dataset_ids_values = [dataset_ids_values]
#                     if len(reader_values) == variables_values:
#                         variables_values
                else:
                    dataset_ids_values = [None]*len(reader_values)

                for option, variables, dataset_ids in zip(reader_values,variables_values,dataset_ids_values):
                    # setup reader with kwargs for that reader
                    # prioritize input kwargs over default args
                    # NEED TO INCLUDE kwargs that go to all the readers
                    args = {}
                    args_in = {**args,
                               **self.kwargs_all,
#                                reader_key: option,
#                                **self.kwargs[source.reader],
                               }

                    if source.reader in self.kwargs.keys():
                        args_in = {**args_in,
                                   **self.kwargs[source.reader],
                                  }

                    args_in = {**args_in,
                              reader_key: option}

                    # deal with variables separately
                    args_in = {**args_in,
                               'variables': variables,
                              }

                    # deal with dataset_ids separately
                    args_in = {**args_in,
                               'dataset_ids': dataset_ids,
                              }


                    if self.kwargs_all['approach'] == 'region':
                        reader = source.region(args_in)
                    elif self.kwargs_all['approach'] == 'stations':
                        reader = source.stations(args_in)


                    sources.append(reader)

            self._sources = sources

        return self._sources


    @property
    def dataset_ids(self):

        if not hasattr(self, '_dataset_ids'):

            dataset_ids = []
            for source in self.sources:

                dataset_ids.append(source.dataset_ids)

            self._dataset_ids = dataset_ids

        return self._dataset_ids


    @property
    def meta(self):
        '''Find and return metadata for datasets.

        Do this by querying each data source function for metadata
        then use the metadata for quick returns.

        This will not rerun if the metadata has already been found.

        SEPARATE DATASOURCE FUNCTIONS INTO A PART THAT RETRIEVES THE
        DATASET_IDS AND METADATA AND A PART THAT READS IN THE DATA.

        DIFFERENT SOURCES HAVE DIFFERENT METADATA

        START EVERYTHING BEING REGION BASED BUT LATER MAYBE ADD A STATION
        OPTION.

        EXPOSE DATASET_IDS?

        '''

        if not hasattr(self, '_meta'):

            # loop over data sources to read in metadata
            meta = []
            for source in self.sources:

                meta.append(source.meta)

            self._meta = meta

        return self._meta


    @property
    def data(self):
        '''Return the data, given metadata.'''

        if not hasattr(self, '_data'):

            # loop over data sources to read in data
            data = []
            for source in self.sources:

                data.append(source.data)

            self._data = data

        return self._data

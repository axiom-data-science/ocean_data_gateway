from erddapy import ERDDAP
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import xarray as xr
import logging
import os
import re
import numpy as np
import pathlib
import ocean_data_gateway as odg


# Capture warnings in log
logging.captureWarnings(True)

# formatting for logfile
formatter = logging.Formatter('%(asctime)s %(message)s','%a %b %d %H:%M:%S %Z %Y')
log_name = 'reader_erddap'
loglevel=logging.WARNING
path_logs_reader = odg.path_logs.joinpath(f'{log_name}.log')

# set up logger file
handler = logging.FileHandler(path_logs_reader)
handler.setFormatter(formatter)
logger_erd = logging.getLogger(log_name)
logger_erd.setLevel(loglevel)
logger_erd.addHandler(handler)

# this can be queried with
# search.ErddapReader.reader
reader = 'erddap'


class ErddapReader:


    def __init__(self, known_server='ioos', protocol=None, server=None, parallel=True):

#         # run checks for KW
#         self.kw = kw

        self.parallel = parallel


        # either select a known server or input protocol and server string
        if known_server == 'ioos':
            protocol = 'tabledap'
            server = 'http://erddap.sensors.ioos.us/erddap'
        elif known_server == 'coastwatch':
            protocol = 'griddap'
            server = 'http://coastwatch.pfeg.noaa.gov/erddap'
        elif known_server is not None:
            statement = 'either select a known server or input protocol and server string'
            assert (protocol is not None) & (server is not None), statement
        else:
            known_server = server.strip('/erddap').strip('http://').replace('.','_')
            statement = 'either select a known server or input protocol and server string'
            assert (protocol is not None) & (server is not None), statement

        self.known_server = known_server
        self.e = ERDDAP(server=server)
        self.e.protocol = protocol
        self.e.server = server

        # columns for metadata
        self.columns = ['geospatial_lat_min', 'geospatial_lat_max',
               'geospatial_lon_min', 'geospatial_lon_max',
               'time_coverage_start', 'time_coverage_end',
               'defaultDataQuery', 'subsetVariables',  # first works for timeseries sensors, 2nd for gliders
               'keywords',  # for hf radar
               'id', 'infoUrl', 'institution', 'featureType', 'source', 'sourceUrl']

        # name
        self.name = f'erddap_{known_server}'

        self.reader = 'ErddapReader'

# #         self.data_type = data_type
#         self.standard_names = standard_names
#         # DOESN'T CURRENTLY LIMIT WHICH VARIABLES WILL BE FOUND ON EACH SERVER



    @property
    def dataset_ids(self):
        '''Find dataset_ids for server.'''

        if not hasattr(self, '_dataset_ids'):

            # This should be a region search
            if self.approach == 'region':

                # find all the dataset ids which we will use to get the data
                # This limits the search to our keyword arguments in kw which should
                # have min/max lon/lat/time values
                dataset_ids = []
                if self.variables is not None:
                    for variable in self.variables:

                        # find and save all dataset_ids associated with variable
                        search_url = self.e.get_search_url(response="csv", **self.kw,
                                                           variableName=variable,
                                                           items_per_page=10000)

                        try:
                            search = pd.read_csv(search_url)
                            dataset_ids.extend(search["Dataset ID"])
                        except Exception as e:
                            logger_erd.exception(e)
                            logger_erd.warning(f"variable {variable} was not found in the search")
                            logger_erd.warning(f'search_url: {search_url}')

                else:

                    # find and save all dataset_ids associated with variable
                    search_url = self.e.get_search_url(response="csv", **self.kw,
                                                       items_per_page=10000)

                    try:
                        search = pd.read_csv(search_url)
                        dataset_ids.extend(search["Dataset ID"])
                    except Exception as e:
                        logger_erd.exception(e)
                        logger_erd.warning(f"nothing found in the search")
                        logger_erd.warning(f'search_url: {search_url}')


                # only need a dataset id once since we will check them each for all standard_names
                self._dataset_ids = list(set(dataset_ids))

            # This should be a search for the station names
            elif self.approach == 'stations':
#             elif self._stations is not None:

                # search by station name for each of stations
                dataset_ids = []
                for station in self._stations:
                    # if station has more than one word, AND will be put between to search for multiple
                    # terms together
                    url = self.e.get_search_url(response="csv", items_per_page=5, search_for=station)

                    try:
                        df = pd.read_csv(url)
                    except Exception as e:
                        logger_erd.exception(e)
                        logger_erd.warning(f'search url {url} did not work for station {station}.')
                        continue

                    # first try for exact station match
                    try:
                        dataset_id = [dataset_id for dataset_id in df['Dataset ID'] if station.lower() in dataset_id.lower().split('_')][0]

                    # if that doesn't work, trying for more general match and just take first returned option
                    except Exception as e:
                        logger_erd.exception(e)
                        logger_erd.warning('When searching for a dataset id to match station name %s, the first attempt to match the id did not work.' % (station))
                        dataset_id = df.iloc[0]['Dataset ID']

#                         if 'tabs' in org_id:  # don't split
#                             axiom_id = [axiom_id for axiom_id in df['Dataset ID'] if org_id.lower() == axiom_id.lower()]
#                         else:
#                             axiom_id = [axiom_id for axiom_id in df['Dataset ID'] if org_id.lower() in axiom_id.lower().split('_')][0]

#                     except:
#                         dataset_id = None

                    dataset_ids.append(dataset_id)

                self._dataset_ids = list(set(dataset_ids))

            else:
                logger_erd.warning('Neither stations nor region approach were used in function dataset_ids.')


        return self._dataset_ids


    def meta_by_dataset(self, dataset_id):

        info_url = self.e.get_info_url(response="csv", dataset_id=dataset_id)
        try:
            info = pd.read_csv(info_url)
        except Exception as e:
            logger_erd.exception(e)
            logger_erd.warning(f'Could not read info from {info_url}')
            return {dataset_id: []}

        items = []

        for col in self.columns:

            try:
                item = info[info['Attribute Name'] == col]['Value'].values[0]
                dtype = info[info['Attribute Name'] == col]['Data Type'].values[0]
            except:
                if col == 'featureType':
                    # this column is not present in HF Radar metadata but want it to
                    # map to data_type, so input 'grid' in that case.
                    item = 'grid'
                else:
                    item = 'NA'

            if dtype == 'String':
                pass
            elif dtype == 'double':
                item = float(item)
            elif dtype == 'int':
                item = int(item)
            items.append(item)

#         if self.standard_names is not None:
#             # In case the variable is named differently from the standard names,
#             # we back out the variable names here for each dataset. This also only
#             # returns those names for which there is data in the dataset.
#             varnames = self.e.get_var_by_attr(
#                 dataset_id=dataset_id,
#                 standard_name=lambda v: v in self.standard_names
#             )
#         else:
#             varnames = None

        ## include download link ##
        self.e.dataset_id = dataset_id
        if self.e.protocol == 'tabledap':
            if self.variables is not None:
                self.e.variables = ["time","longitude", "latitude", "station"] + self.variables
            # set the same time restraints as before
            self.e.constraints = {'time<=': self.kw['max_time'], 'time>=': self.kw['min_time'],}
            download_url = self.e.get_download_url(response='csvp')

        elif self.e.protocol == 'griddap':
            # the search terms that can be input for tabledap do not work for griddap
            # in erddapy currently. Instead, put together an opendap link and then
            # narrow the dataset with xarray.
            # get opendap link
            download_url = self.e.get_download_url(response='opendap')

        # add erddap server name
        return {dataset_id: [self.e.server, download_url] + items + [self.variables]}


    @property
    def meta(self):

        if not hasattr(self, '_meta'):

            if self.parallel:

                # get metadata for datasets
                # run in parallel to save time
                num_cores = multiprocessing.cpu_count()
                downloads = Parallel(n_jobs=num_cores)(
                    delayed(self.meta_by_dataset)(dataset_id) for dataset_id in self.dataset_ids
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
            self._meta = pd.DataFrame.from_dict(meta, orient='index',
                                                columns=['database','download_url'] \
                                                + self.columns + ['variable names'])

        return self._meta


    def data_by_dataset(self, dataset_id):

        download_url = self.meta.loc[dataset_id, 'download_url']
        # data variables in ds that are not the variables we searched for
#         varnames = self.meta.loc[dataset_id, 'variable names']

        if self.e.protocol == 'tabledap':

            try:

                # fetch metadata if not already present
                # found download_url from metadata and use
                dd = pd.read_csv(download_url, index_col=0, parse_dates=True)

                # Drop cols and rows that are only NaNs.
                dd = dd.dropna(axis='index', how='all').dropna(axis='columns', how='all')

                if self.variables is not None:
                    # check to see if there is any actual data
                    # this is a bit convoluted because the column names are the variable names
                    # plus units so can't match 1 to 1.
                    datacols = 0  # number of columns that represent data instead of metadata
                    for col in dd.columns:
                        datacols += [varname in col for varname in self.variables].count(True)
                    # if no datacols, we can skip this one.
                    if datacols == 0:
                        dd = None

            except Exception as e:
                logger_erd.exception(e)
                logger_erd.warning('no data to be read in for %s' % dataset_id)
                dd = None

        elif self.e.protocol == 'griddap':

            try:
                dd = xr.open_dataset(download_url, chunks='auto').sel(time=slice(self.kw['min_time'],self.kw['max_time']))

                if ('min_lat' in self.kw) and ('max_lat' in self.kw):
                    dd = dd.sel(latitude=slice(self.kw['min_lat'],self.kw['max_lat']))

                if ('min_lon' in self.kw) and ('max_lon' in self.kw):
                    dd = dd.sel(longitude=slice(self.kw['min_lon'],self.kw['max_lon']))

                # use variable names to drop other variables (should. Ido this?)
                if self.variables is not None:
                    l = set(dd.data_vars) - set(self.variables)
                    dd = dd.drop_vars(l)

            except Exception as e:
                logger_erd.exception(e)
                logger_erd.warning('no data to be read in for %s' % dataset_id)
                dd = None

        return (dataset_id, dd)


    # @property
    def data(self):

        # if not hasattr(self, '_data'):

        if self.parallel:
            num_cores = multiprocessing.cpu_count()
            downloads = Parallel(n_jobs=num_cores)(
                delayed(self.data_by_dataset)(dataset_id) for dataset_id in self.dataset_ids
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


    def count(self,url):
        try:
            return len(pd.read_csv(url))
        except:
            return np.nan


    def all_variables(self):
        '''Return a list of all possible variables.'''

        path_name_counts = odg.path_variables.joinpath(f'erddap_variable_list_{self.known_server}.csv')

        if path_name_counts.is_file():
            return pd.read_csv(path_name_counts, index_col='variable')
        else:
            print('Please wait while the list of available variables is made. This only happens once but could take 10 minutes.')
            # This took 10 min running in parallel for ioos
            # 2 min for coastwatch
            url = f'{self.e.server}/categorize/variableName/index.csv?page=1&itemsPerPage=100000'
            df = pd.read_csv(url)
#             counts = []
#             for url in df.URL:
#                 counts.append(self.count(url))
            num_cores = multiprocessing.cpu_count()
            counts = Parallel(n_jobs=num_cores)(
                delayed(self.count)(url) for url in df.URL
            )
            dfnew = pd.DataFrame()
            dfnew['variable'] = df['Category']
            dfnew['count'] = counts
            dfnew = dfnew.set_index('variable')
            # remove nans
            if (dfnew.isnull().sum() > 0).values:
                dfnew = dfnew[~dfnew.isnull().values].astype(int)
            dfnew.to_csv(path_name_counts)

        return dfnew


    def search_variables(self, variables):
        '''Find valid variables names to use.

        Call with `search_variables()` to return the list of possible names.
        Call with `search_variables('salinity')` to return relevant names.
        '''

        if not isinstance(variables, list):
            variables = [variables]

        # set up search for input variables
        search = f"(?i)"
        for variable in variables:
            search += f".*{variable}|"
        search = search.strip('|')

        r = re.compile(search)

        # just get the variable names
        df = self.all_variables()
        parameters = df.index

        matches = list(filter(r.match, parameters))

        # return parameters that match input variable strings
        return df.loc[matches].sort_values('count', ascending=False)


    def check_variables(self, variables, verbose=False):

        if not isinstance(variables, list):
            variables = [variables]

#         parameters = list(self.all_variables().keys())
        parameters = list(self.all_variables().index)

        # for a variable to exactly match a parameter
        # this should equal 1
        count = []
        for variable in variables:
            count += [parameters.count(variable)]

        condition = np.allclose(count,1)

        assertion = f'The input variables are not exact matches to ok variables for known_server {self.known_server}. \
                     \nCheck all parameter group values with `ErddapReader().all_variables()` \
                     \nor search parameter group values with `ErddapReader().search_variables({variables})`.\
                     \n\n Try some of the following variables:\n{str(self.search_variables(variables))}'# \
#                      \nor run `ErddapReader().check_variables("{variables}")'
        assert condition, assertion

        if condition and verbose:
            print('all variables are matches!')


    # Search for stations by region
class region(ErddapReader):
#     def region(self, kw, variables=None):

    def __init__(self, kwargs):
        assert isinstance(kwargs, dict), 'input arguments as dictionary'
        er_kwargs = {'known_server': kwargs.get('known_server', 'ioos'),
                     'protocol': kwargs.get('protocol', None),
                     'server': kwargs.get('server', None),
                     'parallel': kwargs.get('parallel', True)}
        ErddapReader.__init__(self, **er_kwargs)

        kw = kwargs['kw']
        variables = kwargs.get('variables', None)

        self.approach = 'region'

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


# #         self.data_type = data_type
#         if not isinstance(standard_names, list):
#             standard_names = [standard_names]
#         self.standard_names = standard_names
    # DOESN'T CURRENTLY LIMIT WHICH VARIABLES WILL BE FOUND ON EACH SERVER

#     return self


class stations(ErddapReader):
#     def stations(self, dataset_ids=None, stations=None, kw=None):
#         '''

#         Use keyword dataset_ids if you already know the database-
#         specific ids. Otherwise, use the keyword stations and the
#         database-specific ids will be searched for. The station
#         ids can be input as something like "TABS B" and will be
#         searched for as "TABS AND B" and has pretty good success.
#         '''


    def __init__(self, kwargs):
        assert isinstance(kwargs, dict), 'input arguments as dictionary'
        er_kwargs = {'known_server': kwargs.get('known_server', 'ioos'),
                     'protocol': kwargs.get('protocol', None),
                     'server': kwargs.get('server', None),
                     'parallel': kwargs.get('parallel', True)}
        ErddapReader.__init__(self, **er_kwargs)

        kw = kwargs.get('kw', None)
        dataset_ids = kwargs.get('dataset_ids', None)
        stations = kwargs.get('stations', [])

        self.approach = 'stations'

        # we want all the data associated with stations
#         self.standard_names = None
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
            kw = {'min_time': '1900-01-01', 'max_time': '2100-12-31'}

        self.kw = kw

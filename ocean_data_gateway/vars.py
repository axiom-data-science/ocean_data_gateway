"""
Functions to check variable names for readers.
"""

import multiprocessing
import os
import re
import urllib.parse

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import ocean_data_gateway as odg


def line_count(url):
    """Small helper function to count len(results) at url."""
    try:
        return len(pd.read_csv(url))
    except:
        return np.nan


def all_variables(server, parallel=True):
    """Return a DataFrame of allowed variable names.

    Parameters
    ----------
    server: str
        Information for the reader, as follows:
        * For an ERDDAP reader, `server` should be the ERDDAP server
          input as a string. For example, http://erddap.sensors.ioos.us/erddap.
        * For the axds reader, `server` should just be 'axds'. Note that
          the variable list is only valid for `axds_type=='platform2'`, not for
          'layer_group'
    parallel: boolean
        If True, run with simple parallelization using `multiprocessing`.
        If False, run serially.

    Returns
    -------
    DataFrame of variable names and count of how many times they are present in the database.

    Notes
    -----
    For an ERDDAP server, the variable list is specific to the given ERDDAP server. If you are using a user-input server, it will have its own `known_server` name and upon running this function the first time, you should get a variable list for that server.
    There is only one variable list for `server=='axds'`.

    Examples
    --------

    For the IOOS ERDDAP server:

    >>> import ocean_data_gateway as odg
    >>> server = 'http://erddap.sensors.ioos.us/erddap'
    >>> odg.all_variables(server=server)
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

    For the Coastwatch ERDDAP server:

    >>> server = 'http://coastwatch.pfeg.noaa.gov/erddap'
    >>> odg.all_variables(server=server)
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

    For axds reader (`axds_type=='platform2'`):

    >>> odg.all_variables(server='axds')
                                                     count
    variable
    Ammonium                                            23
    Atmospheric Pressure: Air Pressure at Sea Level    362
    Atmospheric Pressure: Barometric Pressure         4152
    Backscatter Intensity                              286
    Battery                                           2705
    ...                                                ...
    Winds: Samples                                       1
    Winds: Speed and Direction                        7091
    Winds: Vertical Wind                                 4
    Winds: at 10 m                                      18
    pH                                                 965
    """

    if "axds" in server:

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

    # This is an ERDDAP server
    else:
        server_name = urllib.parse.urlparse(server).netloc
        path_name_counts = odg.variables_path.joinpath(
            f"erddap_variable_list_{server_name}.csv"
        )

        if path_name_counts.is_file():
            return pd.read_csv(path_name_counts, index_col="variable")
        else:
            print(
                "Please wait while the list of available variables is made. This only happens once but could take 10 minutes."
            )
            # This took 10 min running in parallel for ioos
            # 2 min for coastwatch
            url = (
                f"{server}/categorize/variableName/index.csv?page=1&itemsPerPage=100000"
            )
            df = pd.read_csv(url)
            if not parallel:
                counts = []
                for url in df.URL:
                    counts.append(line_count(url))
            else:
                num_cores = multiprocessing.cpu_count()
                counts = Parallel(n_jobs=num_cores)(
                    delayed(line_count)(url) for url in df.URL
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


def search_variables(server, variables):
    """Find valid variables names to use.

    Parameters
    ----------
    server: str
        Information for the reader, as follows:
        * For an ERDDAP reader, `server` should be the ERDDAP server
          input as a string. For example, http://erddap.sensors.ioos.us/erddap.
        * For the axds reader, `server` should just be 'axds'. Note that
          the variable list is only valid for `axds_type=='platform2'`, not for
          'layer_group'
    variables: string, list
        String or list of strings to use in regex search to find valid
        variable names.

    Returns
    -------
    DataFrame of variable names and count of how many times they are present in the database, sorted by count.

    Notes
    -----
    The return list is specific to the ERDDAP server input or to axds data for type 'platform2' (not 'layer_group').

    Examples
    --------

    For the IOOS ERDDAP server:

    Search for variables that contain the substring 'sal':

    >>> server='http://erddap.sensors.ioos.us/erddap'
    >>> odg.search_variables(server=server, variables='sal')
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

    >>> odg.search_variables(server=server, variables='')
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

    For axds reader (`axds_type=='platform2'`):

    Search for variables that contain the substring 'sal':

    >>> odg.search_variables(server='axds', variables='sal')
                   count
    variable
    Salinity        3204
    Soil Salinity    622

    Return all available variables, sorted by count (or could use
    `all_variables()` directly):

    >>> odg.search_variables(server='axds', variables='')
                                                        count
    variable
    Stream Height                                       19758
    Water Surface above Datum                           19489
    Stream Flow                                         15203
    Temperature: Air Temperature                         8369
    Precipitation                                        7364
    ...                                                   ...
    Vent Fluid Temperature                                  1
    Vent Fluid Thermocouple Temperature - Low               1
    CO2: PPM of Carbon Dioxide in Sea Water in Wet Gas      1
    CO2: PPM of Carbon Dioxide in Air in Dry Gas            1
    Evaporation Rate                                        1
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
    df = all_variables(server=server)
    parameters = df.index

    matches = list(filter(r.match, parameters))

    # return parameters that match input variable strings
    return df.loc[matches].sort_values("count", ascending=False)


def check_variables(server, variables, verbose=False):
    """Checks variables for presence in database list.

    Parameters
    ----------
    server: str
        Information for the reader, as follows:
        * For an ERDDAP reader, `server` should be the ERDDAP server
          input as a string. For example, http://erddap.sensors.ioos.us/erddap.
        * For the axds reader, `server` should just be 'axds'. Note that
          the variable list is only valid for `axds_type=='platform2'`, not for
          'layer_group'
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
    if variables is/are valid variable name(s), nothing happens.

    Notes
    -----
    This list is specific to the ERDDAP server being used, or to the axds search for 'platform2' (does not work for `layer_group`).

    Examples
    --------

    For the IOOS ERDDAP server:

    Check if the variable name 'sal' is valid:

    >>> server = 'http://erddap.sensors.ioos.us/erddap'
    >>> odg.check_variables(server, 'sal')
    AssertionError                            Traceback (most recent call last)
    <ipython-input-2-f1434ac5886a> in <module>
          1 server = 'http://erddap.sensors.ioos.us/erddap'
    ----> 2 odg.check_variables(server, 'sal')

    ~/projects/ocean_data_gateway/ocean_data_gateway/vars.py in check_variables(server, variables, verbose)
        404                  \nor search variable names with `odg.search_variables({server}, {variables})`.\
        405                  \n\n Try some of the following variables:\n{str(search_variables(server,variables))}"  # \
    --> 406     assert condition, assertion
        407
        408     if condition and verbose:

    AssertionError: The input variables are not exact matches to variables for server http://erddap.sensors.ioos.us/erddap.
    Check all variable names with `odg.all_variables(server=http://erddap.sensors.ioos.us/erddap)`
    or search variable names with `odg.search_variables(http://erddap.sensors.ioos.us/erddap, ['sal'])`.

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

    >>> odg.check_variables(server, 'salinity')

    For axds reader (`axds_type=='platform2'`):

    Check if the variable name 'sal' is valid:

    >>> odg.check_variables('axds', 'sal')
    AssertionError                            Traceback (most recent call last)
    <ipython-input-2-cb700d016565> in <module>
    ----> 1 odg.check_variables('axds', 'sal')

    ~/projects/ocean_data_gateway/ocean_data_gateway/vars.py in check_variables(server, variables, verbose)
        421                  \nor search variable names with `odg.search_variables({server}, {variables})`.\
        422                  \n\n Try some of the following variables:\n{str(search_variables(server,variables))}"  # \
    --> 423     assert condition, assertion
        424
        425     if condition and verbose:

    AssertionError: The input variables are not exact matches to variables for server axds.
    Check all variable names with `odg.all_variables(server=axds)`
    or search variable names with `odg.search_variables(axds, ['sal'])`.

     Try some of the following variables:
                   count
    variable
    Salinity        3204
    Soil Salinity    622

    Check if the variable name 'Salinity' is valid:

    >>> odg.check_variables('axds', 'Salinity')

    """

    if not isinstance(variables, list):
        variables = [variables]

    parameters = list(all_variables(server).index)

    # for a variable to exactly match a parameter
    # this should equal 1
    count = []
    for variable in variables:
        count += [parameters.count(variable)]

    condition = np.allclose(count, 1)

    assertion = f"The input variables are not exact matches to variables for server {server}. \
                 \nCheck all variable names with `odg.all_variables(server={server})` \
                 \nor search variable names with `odg.search_variables({server}, {variables})`.\
                 \n\n Try some of the following variables:\n{str(search_variables(server,variables))}"  # \
    assert condition, assertion

    if condition and verbose:
        print("all variables are matches!")


def select_variables(server, criteria, variables):
    """Use variable criteria to choose from available variables.

    Parameters
    ----------
    server: str
        Information for the reader, as follows:
        * For an ERDDAP reader, `server` should be the ERDDAP server
          input as a string. For example, http://erddap.sensors.ioos.us/erddap.
        * For the axds reader, `server` should just be 'axds'. Note that
          the variable list is only valid for `axds_type=='platform2'`, not for
          'layer_group'
    criteria: dict, str
        Custom criteria input by user to determine which variables to select.
    variables: string, list
        String or list of strings to compare against list of valid
        variable names. They should be keys in `criteria`.

    Returns
    -------
    Variables from server that match with inputs.

    Notes
    -----
    This uses logic from `cf-xarray`.
    """

    df = all_variables(server)

    results = []
    for key in variables:
        if key in criteria:
            for criterion, patterns in criteria[key].items():
                results.extend(
                    list(set([var for var in df.index if re.match(patterns, var)]))
                )

        # catch scenario that user input valid reader variable names
        else:
            check_variables(server, variables)
            results = variables
    return results

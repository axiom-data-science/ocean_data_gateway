"""
Utilities to help with running the other code.

match_var()
"""

import multiprocessing

# https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
from collections.abc import MutableMapping

import pandas as pd

from joblib import Parallel, delayed


def resample_like(ds_resample, ds_resample_to):
    """Resample input dataset in time like other dataset.

    Parameters
    ----------
    ds_resample: xarray Dataset
        Dataset to resample
    ds_resample_to: xarray Dataset
        Resample ds_resample to be at same times as ds_resample_to

    Returns
    -------
    ds_resample with times matches ds_resample_to.

    TODO: Does this work for upsampling as well as downsampling?
    TODO: Should the index be taken from ds_resample_to and example matched by
    reindexing?
    """

    # Find delta T for model (lower temporal resolution)
    dt = pd.Timestamp(ds_resample_to.cf["T"][1].values) - pd.Timestamp(
        ds_resample_to.cf["T"][0].values
    )
    dt = dt.total_seconds() / 3600  # delta t in hours

    # resample higher res data set to dt
    ds_resampled = ds_resample.resample(time=f"{dt}H").interpolate("linear")

    return ds_resampled


def load_data(self, dataset_ids=None):
    """Read in data for readers some or all dataset_ids.

    NOT USED CURRENTLY

    Once data is read in for a dataset_ids, it is remembered. This is used in
    all of the readers.

    Parameters
    ----------
    dataset_ids: string, list, optional
        Read in data for dataset_ids specifically, or return a look at the
        currently available data. Options are:
        * a string of 1 dataset_id
        * a list of strings of dataset_ids
        * None or nothing entered: data will be read in for all
          `self.dataset_ids`.
        * 'printkeys' will return a dictionary of the dataset_ids already read
          in
        * 'printall' will return all the data available

    Returns
    -------
    There is different behavior for different inputs. If `dataset_ids` is a string, the Dataset for that dataset_id will be returned. If `dataset_ids` is a list of dataset_ids or `dataset_ids==None`, a dictionary will be returned with keys of the dataset_ids and values the data of type pandas DataFrame. Or use the strings 'printkeys' or 'printall' for a look at the data presently available.

    Notes
    -----
    This is either done in parallel with the `multiprocessing` library or
    in serial.
    """

    # first time called, set up self._data as a dict
    if not hasattr(self, "_data"):
        self._data = {}

    if dataset_ids == "printkeys":
        return self._data.keys()

    elif dataset_ids == "printall":
        return self._data

    # for a single dataset_ids, just return that Dataset
    elif (dataset_ids is not None) and (isinstance(dataset_ids, str)):

        assertion = "dataset_id is not valid for this search"
        assert dataset_ids in self.dataset_ids, assertion

        if dataset_ids not in self._data:
            self._data[dataset_ids] = self.data_by_dataset(dataset_ids)
        return self._data[dataset_ids]

    # Read in data for user-input dataset_ids or all dataset_ids
    elif (dataset_ids is None) or (
        (dataset_ids is not None) and (isinstance(dataset_ids, list))
    ):

        if dataset_ids is None:
            dataset_ids_to_use = self.dataset_ids
        else:
            dataset_ids_to_use = dataset_ids

        # first find which dataset_ids_to_use haven't already been read in
        dataset_ids_to_use = [
            dataid for dataid in dataset_ids_to_use if dataid not in self._data
        ]

        if self.parallel:
            num_cores = multiprocessing.cpu_count()
            downloads = Parallel(n_jobs=num_cores)(
                delayed(self.data_by_dataset)(dataid) for dataid in dataset_ids_to_use
            )
            for dataid, download in zip(dataset_ids_to_use, downloads):
                self._data[dataid] = download

        else:
            for dataid in dataset_ids_to_use:
                self._data[dataid] = self.data_by_dataset(dataid)

    return self._data


# dict-like structure for readers to inherit
class Reader(MutableMapping):
    """dict-like reader class.

    This is the base class for all of the readers so they can have methods and
    can also store and access data in a dict-like manner for convenience, which
    simplifies the syntax a lot.
    """

    def __init__(self):
        """Initialize a reader class."""
        self.store = dict()
        # self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        """This method is overwritten by the readers."""
        # this is overwritten by the same method in each reader
        return key

    def __setitem__(self, key, value):
        """Regular dict-like way to store key/value pair."""
        self.store[key] = value

    def __delitem__(self, key):
        """Regular dict-like way to delete key."""
        del self.store[key]

    def __iter__(self):
        """Regular dict-like way to iter over object."""
        return iter(self.store)

    def __len__(self):
        """Regular dict-like way query length of object."""
        return len(self.store)

    def keys(self):
        """Regular dict-like way to return keys."""
        return self.store.keys()

    def values(self):
        """Regular dict-like way to return values."""
        return self.store.values()


def return_response(url):
    """Return response from website."""

    import requests
    import ast

    # For variable identification with cf-xarray
    # custom_criteria to identify variables is saved here
    # https://gist.github.com/kthyng/c3cc27de6b4449e1776ce79215d5e732
    response = requests.get(url)
    return ast.literal_eval(response.text)

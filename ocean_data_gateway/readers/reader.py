#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reader baseclass definition."""
from collections.abc import MutableMapping


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

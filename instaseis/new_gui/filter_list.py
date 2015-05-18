from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore


class Filter(object):
    pass


class FilterList(QtCore.QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super(FilterList, self).__init__(*args, **kwargs)
        self._filters = []

    def add_filter(self, filter):
        if not isinstance(filter, Filter):
            raise ValueError

        self._filters.append(filter)
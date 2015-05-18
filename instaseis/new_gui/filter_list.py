from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore


class Filter(object):
    pass


class FilterItem(object):
    def __init__(self, filter, name):
        if not isinstance(filter, Filter):
            raise ValueError
        self.filter = filter
        self.name = name


class FilterList(QtCore.QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super(FilterList, self).__init__(*args, **kwargs)
        self._filters = []

    def append(self, filter):
        if not isinstance(filter, FilterItem):
            raise ValueError

        self._filters.append(filter)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._filters)

    def data(self, index, role=None):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return self._filters[index.row()].name
        else:
            return None

    def removeRow(self, row, parent=None, *args, **kwargs):
        del self._filters[row]

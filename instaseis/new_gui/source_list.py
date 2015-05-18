from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore

import instaseis


class SourceItem(object):
    def __init__(self, source, name):
        if not isinstance(source, instaseis.Source):
            raise ValueError
        self.source = source
        self.name = name


class SourceList(QtCore.QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super(SourceList, self).__init__(*args, **kwargs)
        self._sources = []

    def append(self, source):
        if not isinstance(source, SourceItem):
            raise ValueError

        self._sources.append(source)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._sources)

    def data(self, index, role=None):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return self._sources[index.row()].name
        else:
            return None

    def removeRow(self, row, parent=None, *args, **kwargs):
        del self._sources[row]
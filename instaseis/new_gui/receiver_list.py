from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore

import instaseis


class ReceiverItem(object):
    def __init__(self, receiver, name):
        if not isinstance(receiver, instaseis.Receiver):
            raise ValueError
        self.receiver = receiver
        self.name = name


class ReceiverList(QtCore.QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super(ReceiverList, self).__init__(*args, **kwargs)
        self._receivers = []

    def append(self, receiver):
        if not isinstance(receiver, ReceiverItem):
            raise ValueError

        self._receivers.append(receiver)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._receivers)

    def data(self, index, role=None):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return self._receivers[index.row()].name
        else:
            return None

    def removeRow(self, row, parent=None, *args, **kwargs):
        del self._receivers[row]

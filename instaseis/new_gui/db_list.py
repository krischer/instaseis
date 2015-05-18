from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore

import instaseis

class DBItem(object):
    def __init__(self, db, name):
        if not isinstance(db, instaseis.base_instaseis_db.BaseInstaseisDB):
            raise ValueError
        self.db = db
        self.name = name

class DBList(QtCore.QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super(DBList, self).__init__(*args, **kwargs)
        self._dbs = []

    def append(self, db):
        print(db)
        if not isinstance(db, DBItem):
            raise ValueError

        self._dbs.append(db)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._dbs)

    def data(self, index, role=None):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return self._dbs[index.row()].name
        else:
            return None

    def removeRow(self, row, parent=None, *args, **kwargs):
        del self._dbs[row]

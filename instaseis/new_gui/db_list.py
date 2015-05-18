from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore

import instaseis


class DBList(QtCore.QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super(DBList, self).__init__(*args, **kwargs)
        self._dbs = []

    def add_db(self, db):
        if not isinstance(db, instaseis.base_instaseis_db.BaseInstaseisDB):
            raise ValueError

        self._dbs.append(db)

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore, QtGui


class Seismogram(object):
    def __init__(self, db, source, receiver, filter, name):
        self.db = db
        self.source = source
        self.receiver = receiver
        self.filter = filter
        self.name = name


class SeismogramWidget(QtGui.QWidget):
    def __init__(self, name, parent=None):
        super(SeismogramWidget, self).__init__(parent)

        self.title = QtGui.QLabel(name)
        self.title.setStyleSheet('''
            color: rgb(0, 0, 0);
            font-weight: 900;
        ''')

        self.vbox = QtGui.QVBoxLayout()
        self.name_vbox = QtGui.QVBoxLayout()
        self.dropdown_vbox = QtGui.QVBoxLayout()

        items = [
            ("DB:", ["DB_1", "DB_2"]),
            ("Source:", ["asdf_1", "asdklfjadks"]),
            ("Receiver:", ["DB_1", "asdklfjadks"]),
            ("Filter:", ["asdklfjadks", "DB_2"]),
            ]

        for label, content in items:
            _name = QtGui.QLabel(label)
            dropdown = QtGui.QComboBox()
            dropdown.addItems(content)

            self.name_vbox.addWidget(_name)
            self.dropdown_vbox.addWidget(dropdown)

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addItem(self.name_vbox)
        self.hbox.addItem(self.dropdown_vbox)

        self.vbox.addWidget(self.title)
        self.vbox.addItem(self.hbox)

        self.setLayout(self.vbox)



class SeismogramManager(QtCore.QAbstractListModel):
    def __init__(self, db_list, source_list, receiver_list, filter_list,
                 *args, **kwargs):
        super(SeismogramManager, self).__init__(*args, **kwargs)
        self.db_list = db_list
        self.source_list = source_list
        self.receiver_list = receiver_list
        self.filter_list = filter_list
        self._seismograms = []

    def add_seismogram(self, name, db, source, receiver, filter):
        if db not in self.db_list:
            raise ValueError

        if source not in self.source_list:
            raise ValueError

        if receiver not in self.receiver_list:
            raise ValueError

        if filter not in self.filter_list:
            raise ValueError

        self._seismograms.append(Seismogram(db=db, source=source,
                                            receiver=receiver, filter=filter,
                                            name=name))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._seismograms)

    def data(self, index, role=None):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return self._seismograms[index.row()].name
        else:
            return None

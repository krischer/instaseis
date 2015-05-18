import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from PyQt4 import QtGui

from db_list import DBList, DBItem
from source_list import SourceList, SourceItem
from receiver_list import ReceiverList, ReceiverItem
from filter_list import FilterList, FilterItem, Filter

import instaseis
import random


db_list = DBList()

for i in range(2):
    db = instaseis.open_db('/media/ex/instaseis/Martin/DWTh2Ref2_nocrust_50s')
    item = DBItem(db, "db_%i" % i)
    db_list.append(item)


source_list = SourceList()

for i in range(5):
    src = instaseis.Source(latitude=random.random(),
                           longitude=random.random(),
                           depth_in_m=10)
    item = SourceItem(src, "Event_%i" % i)
    source_list.append(item)


receiver_list = ReceiverList()

for i in range(2):
    rec = instaseis.Receiver(latitude=random.random(),
                             longitude=random.random())
    item = ReceiverItem(rec, "rec_%i" % i)
    receiver_list.append(item)


filter_list = FilterList()

for i in range(3):
    filter = Filter()
    item = FilterItem(filter, "filter_%i" % i)
    filter_list.append(item)


class MyWindow(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)

        source_list = SourceList()

        for i in range(5):
            src = instaseis.Source(latitude=random.random(),
                                   longitude=random.random(),
                                   depth_in_m=10)
            item = SourceItem(src, "Event_%i" % i)
            source_list.append(item)

        lv = QListView()
        lv.setModel(source_list)

        # layout
        layout = QVBoxLayout()
        layout.addWidget(lv)
        self.setLayout(layout)


class QCustomQWidget(QtGui.QWidget):
    def __init__(self, name, parent=None):
        super(QCustomQWidget, self).__init__(parent)

        self.title = QtGui.QLabel(name)
        self.title.setStyleSheet('''
            color: rgb(0, 0, 0);
            font-weight: 900;
        ''')

        self.vbox = QtGui.QVBoxLayout()
        self.name_vbox = QtGui.QVBoxLayout()
        self.dropdown_vbox = QtGui.QVBoxLayout()

        labels = ["DB:", "Source:", "Receiver:", "Filter:"]

        for label in labels:
            _name = QtGui.QLabel(label)
            dropdown = QtGui.QComboBox()
            if label == "DB:":
                dropdown.setModel(db_list)
            elif label == "Receiver:":
                dropdown.setModel(receiver_list)
            elif label == "Source:":
                dropdown.setModel(source_list)
            elif label == "Filter:":
                dropdown.setModel(filter_list)
            else:
                raise ValueError

            self.name_vbox.addWidget(_name)
            self.dropdown_vbox.addWidget(dropdown)

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addItem(self.name_vbox)
        self.hbox.addItem(self.dropdown_vbox)

        self.vbox.addWidget(self.title)
        self.vbox.addItem(self.hbox)

        self.setLayout(self.vbox)

class exampleQMainWindow(QtGui.QMainWindow):
    def __init__ (self):
        super(exampleQMainWindow, self).__init__()
        # Create QListWidget
        self.list_widget = QtGui.QListWidget(self)

        for index, name, icon in [
            ('No.1', 'Meyoko',  'icon.png'),
            ('No.2', 'Nyaruko', 'icon.png'),
            ('No.3', 'Louise',  'icon.png')]:
            # Create QCustomQWidget
            custom_widget = QCustomQWidget(name=name)

            original_widget_item = QtGui.QListWidgetItem()
            # Set size hint
            original_widget_item.setSizeHint(custom_widget.sizeHint())

            # Add QListWidgetItem into QListWidget
            self.list_widget.addItem(original_widget_item)
            self.list_widget.setItemWidget(original_widget_item, custom_widget)

        self.setCentralWidget(self.list_widget)


def main():
    app = QApplication(sys.argv)
    # w = MyWindow()
    w = exampleQMainWindow()
    w.show()
    w.raise_()
    sys.exit(app.exec_())





if __name__ == "__main__":
    main()

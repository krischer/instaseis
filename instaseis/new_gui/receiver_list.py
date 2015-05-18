from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from PyQt4 import QtCore

import instaseis


class ReceiverList(QtCore.QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super(ReceiverList, self).__init__(*args, **kwargs)
        self._receivers = []

    def add_receiver(self, receiver):
        if not isinstance(receiver, instaseis.Receiver):
            raise ValueError

        self._receivers.append(receiver)
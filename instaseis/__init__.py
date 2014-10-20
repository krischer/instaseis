#!/usr/bin/env python
from __future__ import absolute_import

import inspect
import os

from .instaseisdb import InstaSeisDB  # NoQa
from .source import Source, Receiver, ForceSource, ReceiverParseError  # NoQa


def test():
    import pytest
    PATH = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    pytest.main(PATH)
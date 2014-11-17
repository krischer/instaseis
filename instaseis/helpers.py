#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Library loading helper.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ctypes as C
import inspect
import os


LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "lib")

cache = []


def load_lib():
    if cache:
        return cache[0]
    else:
        filename = os.path.join(LIB_DIR, "instaseis.so")
        lib = C.CDLL(filename)
        cache.append(lib)
        return lib

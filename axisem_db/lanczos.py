#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers using ctypes around fortran code for Lanczos resampling

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import numpy as np
import ctypes as C

from .helpers import load_lib

lib = load_lib()


def lanczos_resamp(si, dt_old, dt_new, a):
    si = np.require(si, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    n_old = len(si)
    n_new = int(n_old * dt_old / dt_new)
    dt = dt_new / dt_old

    so = np.zeros(n_new, dtype="float64", order="F")

    lib.lanczos_resamp(
        si.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(n_old),
        so.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(n_new),
        C.c_double(dt),
        C.c_int(a))

    return so


def lanczos_kern(x, a):

    kern = C.c_double(0.0)
    lib.lanczos_kern(
        C.c_double(x),
        C.c_int(a),
        C.byref(kern))

    return kern.value

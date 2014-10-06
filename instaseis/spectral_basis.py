#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers using ctypes around some functions from the spectral_basis module from
AxiSEM's kernel module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import ctypes as C
import numpy as np

from .helpers import load_lib


lib = load_lib()


def zelegl(n):
    data = np.zeros(n + 1, np.float64, order="F")
    lib.zelegl(C.c_int(n), data.ctypes.data_as(C.POINTER(C.c_double)))
    return data


def zemngl2(n):
    data = np.zeros(n + 1, np.float64, order="F")
    lib.zemngl2(C.c_int(n), data.ctypes.data_as(C.POINTER(C.c_double)))
    return data


def def_lagrange_derivs_gll(npol):
    G = np.zeros((npol + 1, npol + 1), order="F")
    lib.def_lagrange_derivs_gll(C.c_int(npol),
                                G.ctypes.data_as(C.POINTER(C.c_double)))
    return G


def def_lagrange_derivs_glj(npol):
    G0 = np.zeros(npol + 1, order="F")
    G1 = np.zeros((npol + 1, npol + 1), order="F")
    lib.def_lagrange_derivs_gll(C.c_int(npol),
                                G1.ctypes.data_as(C.POINTER(C.c_double)),
                                G0.ctypes.data_as(C.POINTER(C.c_double)))
    return G0, G1


def lagrange_interpol_2D_td(points1, points2, coefficients, x1, x2):
    points1 = np.require(points1, dtype=np.float64,
                         requirements=["F_CONTIGUOUS"])
    points2 = np.require(points2, dtype=np.float64,
                         requirements=["F_CONTIGUOUS"])
    coefficients = np.require(coefficients, dtype=np.float64,
                              requirements=["F_CONTIGUOUS"])

    if len(points1) != len(points2):
        raise ValueError

    N = len(points1) - 1
    nsamp = coefficients.shape[0]

    interpolant = np.zeros(nsamp, dtype="float64", order="F")

    lib.lagrange_interpol_2D_td(
        C.c_int(N),
        C.c_int(nsamp),
        points1.ctypes.data_as(C.POINTER(C.c_double)),
        points2.ctypes.data_as(C.POINTER(C.c_double)),
        coefficients.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(x1),
        C.c_double(x2),
        interpolant.ctypes.data_as(C.POINTER(C.c_double)))
    return interpolant

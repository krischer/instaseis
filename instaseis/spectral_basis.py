#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers using ctypes around some functions from the spectral_basis module from
AxiSEM's kernel module.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
    Martin van Driel (Martin@vanDriel.de), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import ctypes as C
import numpy as np

from .helpers import load_lib


lib = load_lib()


def lagrange_interpol_2D_td(points1, points2, coefficients, x1, x2):  # NOQA
    points1 = np.require(
        points1, dtype=np.float64, requirements=["F_CONTIGUOUS"]
    )
    points2 = np.require(
        points2, dtype=np.float64, requirements=["F_CONTIGUOUS"]
    )
    coefficients = np.require(
        coefficients, dtype=np.float64, requirements=["F_CONTIGUOUS"]
    )

    # Should be safe enough. This was never raised while extracting a lot of
    # seismograms.
    assert len(points1) == len(points2)

    n = len(points1) - 1
    nsamp = coefficients.shape[0]

    interpolant = np.zeros(nsamp, dtype="float64", order="F")

    lib.lagrange_interpol_2D_td(
        C.c_int(n),
        C.c_int(nsamp),
        points1.ctypes.data_as(C.POINTER(C.c_double)),
        points2.ctypes.data_as(C.POINTER(C.c_double)),
        coefficients.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(x1),
        C.c_double(x2),
        interpolant.ctypes.data_as(C.POINTER(C.c_double)),
    )
    return interpolant

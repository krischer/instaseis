#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers using ctypes around functions in the sem_derivatives module from the
AxiSEM kernel module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ctypes as C
import numpy as np

from .helpers import load_lib


lib = load_lib()


def _strain_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type, axial,
               fct):
    strain_tensor = np.zeros((nsamp, npol + 1, npol + 1, 6), np.float64,
                             order="F")
    u = np.require(u, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    G = np.require(G, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    GT = np.require(GT, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    xi = np.require(xi, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    eta = np.require(eta, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    nodes = np.require(nodes, dtype=np.float64, requirements=["F_CONTIGUOUS"])

    fct(
        u.ctypes.data_as(C.POINTER(C.c_double)),
        G.ctypes.data_as(C.POINTER(C.c_double)),
        GT.ctypes.data_as(C.POINTER(C.c_double)),
        xi.ctypes.data_as(C.POINTER(C.c_double)),
        eta.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(npol),
        C.c_int(nsamp),
        nodes.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(element_type),
        C.c_bool(axial),
        strain_tensor.ctypes.data_as(C.POINTER(C.c_double)))

    return strain_tensor


def strain_monopole_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type,
                       axial):
    return _strain_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type,
                      axial, lib.strain_monopole_td)


def strain_dipole_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type,
                     axial):
    return _strain_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type,
                      axial, lib.strain_dipole_td)


def strain_quadpole_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type,
                       axial):  # pragma: no cover
    return _strain_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type,
                      axial, lib.strain_quadpole_td)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers around some functions from the finite_elem_mapping module from
AxiSEM's kernel module.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import ctypes as C
import numpy as np

from .helpers import load_lib


lib = load_lib()


def inside_element(s, z, nodes, element_type, tolerance):
    in_element = C.c_bool(False)
    xi = C.c_double(0.0)
    eta = C.c_double(0.0)
    nodes = np.require(nodes, requirements=["F_CONTIGUOUS"])
    lib.inside_element(
        C.c_double(s),
        C.c_double(z),
        nodes.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(element_type),
        C.c_double(float(tolerance)),
        C.byref(in_element),
        C.byref(xi),
        C.byref(eta),
    )

    return in_element.value, xi.value, eta.value

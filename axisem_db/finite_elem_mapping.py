#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers around some functions from the finite_elem_mapping module from
AxiSEM's kernel module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import ctypes as C
import numpy as np

from helpers import load_lib


lib = load_lib()


def inside_element(s, z, nodes, element_type, tolerance):
    in_element = C.c_bool(False)
    xi = C.c_double(0.0)
    eta = C.c_double(0.0)
    nodes = np.require(nodes, requirements=["F_CONTIGUOUS"])
    lib.inside_element(C.c_double(s), C.c_double(z),
                       nodes.ctypes.data_as(C.POINTER(C.c_double)),
                       C.c_int(element_type), C.c_double(float(tolerance)),
                       C.byref(in_element), C.byref(xi), C.byref(eta))

    return in_element.value, xi.value, eta.value


def test_inside_element():
    nodes = np.array([
        [4668274.5, 4313461.5],
        [4703863.5, 4274623.],
        [4714964.5, 4284711.],
        [4679291.5,  4323641.]], dtype=np.float64)

    is_in, xi, eta = inside_element(
        s=4676105.76848,
        z=4309398.54759,
        nodes=nodes, element_type=0, tolerance=1E-3)

    assert is_in
    assert abs(xi - -0.68507753579755248 < 1E-5)
    assert abs(eta - -0.60000654152462352 < 1E-5)

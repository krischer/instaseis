#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mostly testing the various wrappers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import numpy as np


from instaseis import finite_elem_mapping, rotations


def test_rotate_frame_rd():
    s, phi, z = rotations.rotate_frame_rd(
        x=9988.6897343821470, y=0.0, z=6358992.1548998145, phi=74.494,
        theta=47.3609999)
    assert abs(s - 4676105.76848060) < 1E-2
    assert abs(phi - 3.14365101866993) < 1E-5
    assert abs(z - 4309398.5475913) < 1E-2


def test_inside_element():
    nodes = np.array([
        [4668274.5, 4313461.5],
        [4703863.5, 4274623.],
        [4714964.5, 4284711.],
        [4679291.5,  4323641.]], dtype=np.float64)

    is_in, xi, eta = finite_elem_mapping.inside_element(
        s=4676105.76848,
        z=4309398.54759,
        nodes=nodes, element_type=0, tolerance=1E-3)

    assert is_in
    assert abs(xi - -0.68507753579755248 < 1E-5)
    assert abs(eta - -0.60000654152462352 < 1E-5)

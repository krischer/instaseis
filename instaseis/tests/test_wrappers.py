#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mostly testing the various wrappers.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import numpy as np


from instaseis import finite_elem_mapping, rotations


def test_rotate_frame_rd():
    s, phi, z = rotations.rotate_frame_rd(
        x=9988.6897343821470,
        y=0.0,
        z=6358992.1548998145,
        phi=74.494,
        theta=47.3609999,
    )
    assert abs(s - 4676105.76848060) < 1e-2
    assert abs(phi - 3.14365101866993) < 1e-5
    assert abs(z - 4309398.5475913) < 1e-2


def test_inside_element():
    nodes = np.array(
        [
            [4668274.5, 4313461.5],
            [4703863.5, 4274623.0],
            [4714964.5, 4284711.0],
            [4679291.5, 4323641.0],
        ],
        dtype=np.float64,
    )

    is_in, xi, eta = finite_elem_mapping.inside_element(
        s=4676105.76848,
        z=4309398.54759,
        nodes=nodes,
        element_type=0,
        tolerance=1e-3,
    )

    assert is_in
    assert abs(xi - -0.68507753579755248) < 1e-5
    assert abs(eta - -0.60000654152462352) < 1e-5

    def check_is_in(
        s,
        z,
        nodes,
        element_type,
        tolerance=1e-3,
        is_in_ref=None,
        xi_ref=None,
        eta_ref=None,
    ):
        is_in, xi, eta = finite_elem_mapping.inside_element(
            s=s,
            z=z,
            nodes=nodes,
            element_type=element_type,
            tolerance=tolerance,
        )

        if is_in_ref is not None:
            assert is_in == is_in_ref
        if xi_ref is not None:
            assert abs(xi - xi_ref) < 1e-5
        if eta_ref is not None:
            assert abs(eta - eta_ref) < 1e-5

    # translated from kerner
    nodes = np.array(
        [[0.0, 1.0], [1.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=np.float64
    )

    # some random location inside the element
    sr = 0.18291
    zr = 1.1456

    # spheroidal
    s0 = 2.0 / 2 ** 0.5
    z0 = s0
    check_is_in(
        nodes=nodes,
        element_type=0,
        s=s0,
        z=z0,
        xi_ref=0,
        eta_ref=0,
        is_in_ref=True,
    )
    check_is_in(nodes=nodes, element_type=0, s=0.5, z=0.5, is_in_ref=False)
    check_is_in(
        nodes=nodes, element_type=0, s=0.5 ** 0.5, z=0.5 ** 0.5, is_in_ref=True
    )
    check_is_in(
        nodes=nodes,
        element_type=0,
        s=3 * 0.5 ** 0.5,
        z=3 * 0.5 ** 0.5,
        is_in_ref=True,
    )
    check_is_in(
        nodes=nodes,
        element_type=0,
        s=sr,
        z=zr,
        is_in_ref=True,
        xi_ref=-0.7984121667508236,
        eta_ref=-0.8398899069206196,
    )

    # subpar
    check_is_in(
        nodes=nodes,
        element_type=1,
        s=1.0,
        z=1.0,
        xi_ref=0,
        eta_ref=0,
        is_in_ref=True,
    )
    check_is_in(nodes=nodes, element_type=1, s=0.5, z=0.5, is_in_ref=True)
    check_is_in(
        nodes=nodes, element_type=1, s=0.5 ** 0.5, z=0.5 ** 0.5, is_in_ref=True
    )
    check_is_in(
        nodes=nodes,
        element_type=1,
        s=3 * 0.5 ** 0.5,
        z=3 * 0.5 ** 0.5,
        is_in_ref=False,
    )
    check_is_in(
        nodes=nodes,
        element_type=1,
        s=sr,
        z=zr,
        is_in_ref=True,
        xi_ref=-0.7246388811525695,
        eta_ref=-0.6714899999999999,
    )

    # semino
    s2 = 1.3106601717798214
    z2 = s2
    check_is_in(
        nodes=nodes,
        element_type=2,
        s=s2,
        z=z2,
        xi_ref=0,
        eta_ref=0,
        is_in_ref=True,
    )
    check_is_in(nodes=nodes, element_type=2, s=0.5, z=0.5, is_in_ref=True)
    check_is_in(
        nodes=nodes, element_type=2, s=0.5 ** 0.5, z=0.5 ** 0.5, is_in_ref=True
    )
    check_is_in(
        nodes=nodes,
        element_type=2,
        s=3 * 0.5 ** 0.5,
        z=3 * 0.5 ** 0.5,
        is_in_ref=True,
    )
    check_is_in(
        nodes=nodes,
        element_type=2,
        s=sr,
        z=zr,
        is_in_ref=True,
        xi_ref=-0.7527609192042671,
        eta_ref=-0.7395369211576499,
    )

    # semiso
    s2 = 1.1035533905932737
    z2 = s2
    check_is_in(
        nodes=nodes,
        element_type=3,
        s=s2,
        z=z2,
        xi_ref=0,
        eta_ref=0,
        is_in_ref=True,
    )
    check_is_in(nodes=nodes, element_type=3, s=0.5, z=0.5, is_in_ref=False)
    check_is_in(
        nodes=nodes, element_type=3, s=0.5 ** 0.5, z=0.5 ** 0.5, is_in_ref=True
    )
    check_is_in(
        nodes=nodes,
        element_type=3,
        s=3 * 0.5 ** 0.5,
        z=3 * 0.5 ** 0.5,
        is_in_ref=False,
    )
    check_is_in(
        nodes=nodes,
        element_type=3,
        s=sr,
        z=zr,
        is_in_ref=True,
        xi_ref=-0.7846998127497518,
        eta_ref=-0.8109601156061497,
    )

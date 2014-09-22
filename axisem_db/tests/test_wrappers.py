#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mostly testing the various wrappers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import numpy as np


from .. import finite_elem_mapping
from .. import rotations
from .. import spectral_basis


def test_rotate_frame_rd():
    s, phi, z = rotations.rotate_frame_rd(
        x=9988.6897343821470, y=0.0, z=6358992.1548998145, phi=74.494,
        theta=47.3609999)
    assert abs(s - 4676105.76848060) < 1E-2
    assert abs(phi - 3.14365101866993) < 1E-5
    assert abs(z - 4309398.5475913) < 1E-2


#def test_azim_factor_bw():
#    factor = rotations.azim_factor_bw(3.143651018669930,
#                                      np.array([0.0, 1.0, 0.0]), 2, 1)
#    assert abs(factor - -0.99999788156734637) < 1E-7


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


def test_zelegl():
    out = spectral_basis.zelegl(4)
    np.testing.assert_allclose(
        out,
        np.array([-1.0, -0.65465367070797720, 0.0, 0.65465367070797720, 1.00]))


def test_zemngl2():
    out = spectral_basis.zemngl2(4)
    np.testing.assert_allclose(
        out,
        np.array([-1.0, -0.507787629, 0.13230082, 0.70882014, 1.0]))


def test_def_lagrange_derivs_gll():
    G = spectral_basis.def_lagrange_derivs_gll(4)
    assert G.shape == (5, 5)
    np.testing.assert_allclose(
        G[:, 0],
        np.array([-5.000, 6.75650248, -2.66666666, 1.4101641, -0.5]))


def test_def_lagrange_derivs_glj():
    G0, G1 = spectral_basis.def_lagrange_derivs_glj(4)
    assert G0.shape == (5,)
    assert G1.shape == (5, 5)
    np.testing.assert_allclose(
        G1[:, 0],
        np.array([-5.000, 6.75650248, -2.66666666, 1.4101641, -0.5]))

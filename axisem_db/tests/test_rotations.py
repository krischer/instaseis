#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing rotation module

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import numpy as np


from .. import rotations


def test_rotate_vector_src_to_xyz():
    # identity
    v = np.array([1., 2., 3.])
    phi = np.radians(0.)
    w = rotations.rotate_vector_src_to_xyz(v, phi)
    wref = v.copy()
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # 180 degrees
    v = np.array([1., 2., 3.])
    phi = np.radians(180.)
    w = rotations.rotate_vector_src_to_xyz(v, phi)
    wref = np.array([-1., -2., 3.])
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # 90 degrees
    v = np.array([1., 2., 3.])
    phi = np.radians(90.)
    w = rotations.rotate_vector_src_to_xyz(v, phi)
    wref = np.array([-2., 1., 3.])
    np.testing.assert_allclose(wref, w, atol=1e-10)


def test_rotate_vector_xyz_to_src():
    # identity
    v = np.array([1., 2., 3.])
    phi = np.radians(0.)
    w = rotations.rotate_vector_xyz_to_src(v, phi)
    wref = v.copy()
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # 180 degrees
    v = np.array([1., 2., 3.])
    phi = np.radians(180.)
    w = rotations.rotate_vector_xyz_to_src(v, phi)
    wref = np.array([-1., -2., 3.])
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # 90 degrees
    v = np.array([1., 2., 3.])
    phi = np.radians(90.)
    w = rotations.rotate_vector_xyz_to_src(v, phi)
    wref = np.array([2., -1., 3.])
    np.testing.assert_allclose(wref, w, atol=1e-10)


def test_rotate_vector_xyz_src_to_xyz_earth():
    # identity
    v = np.array([1., 2., 3.])
    phi = np.radians(0.)
    theta = np.radians(0.)
    w = rotations.rotate_vector_xyz_src_to_xyz_earth(v, phi, theta)
    wref = v.copy()
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # theta = 90, phi = 0
    v = np.array([1., 2., 3.])
    phi = np.radians(0.)
    theta = np.radians(90.)
    w = rotations.rotate_vector_xyz_src_to_xyz_earth(v, phi, theta)
    wref = np.array([3., 2., -1.])
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # theta = 0, phi = 90
    v = np.array([1., 2., 3.])
    phi = np.radians(90.)
    theta = np.radians(0.)
    w = rotations.rotate_vector_xyz_src_to_xyz_earth(v, phi, theta)
    wref = np.array([-2., 1., 3.])
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # theta = 90, phi = 90
    v = np.array([1., 2., 3.])
    phi = np.radians(90.)
    theta = np.radians(90.)
    w = rotations.rotate_vector_xyz_src_to_xyz_earth(v, phi, theta)
    wref = np.array([-2., 3., -1.])
    np.testing.assert_allclose(wref, w, atol=1e-10)


def test_rotate_vector_xyz_earth_to_xyz_src():
    # identity
    v = np.array([1., 2., 3.])
    phi = np.radians(0.)
    theta = np.radians(0.)
    w = rotations.rotate_vector_xyz_earth_to_xyz_src(v, phi, theta)
    wref = v.copy()
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # theta = 90, phi = 0
    v = np.array([1., 2., 3.])
    phi = np.radians(0.)
    theta = np.radians(90.)
    w = rotations.rotate_vector_xyz_earth_to_xyz_src(v, phi, theta)
    wref = np.array([-3., 2., 1.])
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # theta = 0, phi = 90
    v = np.array([1., 2., 3.])
    phi = np.radians(90.)
    theta = np.radians(0.)
    w = rotations.rotate_vector_xyz_earth_to_xyz_src(v, phi, theta)
    wref = np.array([2., -1., 3.])
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # theta = 90, phi = 90
    v = np.array([1., 2., 3.])
    phi = np.radians(90.)
    theta = np.radians(90.)
    w = rotations.rotate_vector_xyz_earth_to_xyz_src(v, phi, theta)
    wref = np.array([-3., -1., 2.])
    np.testing.assert_allclose(wref, w, atol=1e-10)

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


def test_rotate_tensor_xyz_earth_to_xyz_src():
    mt = np.array([1., 2., 3., 4., 5., 6.])
    phi = np.radians(13.)
    theta = np.radians(29.)
    mt_rot = rotations.rotate_symm_tensor_voigt_xyz_earth_to_xyz_src(
        mt, phi, theta)
    mt_ref = np.array([
        -1.37383329, -0.68082986, 8.05466315,
        5.14580719, 3.34719916, 3.56407819])
    np.testing.assert_allclose(mt_ref, mt_rot, atol=1e-10)


def test_rotate_tensor_xyz_src_to_xyz_earth():
    mt = np.array([1., 2., 3., 4., 5., 6.])
    phi = np.radians(13.)
    theta = np.radians(29.)
    mt_rot = rotations.rotate_symm_tensor_voigt_xyz_src_to_xyz_earth(
        mt, phi, theta)
    mt_ref = np.array([
        2.37201346, 5.33830776, -1.71032122,
        1.36130796, 3.27536413, 7.2728428])
    np.testing.assert_allclose(mt_ref, mt_rot, atol=1e-10)


def test_rotate_tensor_xyz_to_src():
    mt = np.array([1., 2., 3., 4., 5., 6.])
    phi = np.radians(13.)
    mt_rot = rotations.rotate_symm_tensor_voigt_xyz_to_src(mt, phi)
    mt_ref = np.array([
        3.68082986, -0.68082986, 3.,
        2.77272499, 5.77165454, 5.61194985])
    np.testing.assert_allclose(mt_ref, mt_rot, atol=1e-10)


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

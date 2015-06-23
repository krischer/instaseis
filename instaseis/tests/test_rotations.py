#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing rotation module

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import numpy as np


from instaseis import rotations


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


def test_rotate_vector_xyz_src_to_xyz_rec():
    # identity
    v = np.array([1., 2., 3.])
    phi1 = np.radians(20.)
    theta1 = np.radians(30.)
    phi2, theta2 = phi1, theta1
    w = rotations.rotate_vector_xyz_src_to_xyz_rec(
        v, phi1, theta1, phi2, theta2)
    wref = v.copy()
    np.testing.assert_allclose(wref, w, atol=1e-10)

    # theta1 = 0, phi1 = 90, theta2 = 0, phi2 = -90
    v = np.array([1., 2., 3.])
    phi1 = np.radians(90.0)
    theta1 = np.radians(0.0)
    phi2 = np.radians(-90.0)
    theta2 = np.radians(0.0)
    w = rotations.rotate_vector_xyz_src_to_xyz_rec(
        v, phi1, theta1, phi2, theta2)
    wref = np.array([-1., -2., 3.])
    np.testing.assert_allclose(wref, w, atol=1e-10)


def test_coord_transform_lat_lon_depth_to_xyz():
    latitude, longitude, depth_in_m = 0., 0., 0.
    xyz = rotations.coord_transform_lat_lon_depth_to_xyz(
        latitude, longitude, depth_in_m, planet_radius=6371e3)

    np.testing.assert_allclose(np.array([6371e3, 0., 0.]), xyz)


def test_coord_transform_lat_lon_depth_to_xyz_2():
    latitude, longitude, depth_in_m = 23., 32., 1100.
    xyz = rotations.coord_transform_lat_lon_depth_to_xyz(
        latitude, longitude, depth_in_m, planet_radius=6371e3)

    lat, lon, dep = rotations.coord_transform_xyz_to_lat_lon_depth(
        xyz[0], xyz[1], xyz[2], planet_radius=6371e3)

    np.testing.assert_allclose(np.array([latitude, longitude, depth_in_m]),
                               np.array([lat, lon, dep]))

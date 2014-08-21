#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions dealing with rotations. Mostly wrappers using ctypes around Fortran
code from the AxiSEM kernel module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import ctypes as C
import numpy as np

from .helpers import load_lib


lib = load_lib()
lib.azim_factor_bw.restype = C.c_double


def rotate_frame_rd(x, y, z, phi, theta):
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    # first rotation (longitude)
    xp_cp = x * np.cos(phi) + y * np.sin(phi)
    yp_cp = -x * np.sin(phi) + y * np.cos(phi)
    zp_cp = z

    # second rotation (colat)
    xp = xp_cp * np.cos(theta) - zp_cp * np.sin(theta)
    yp = yp_cp
    zp = xp_cp * np.sin(theta) + zp_cp * np.cos(theta)

    srd = np.sqrt(xp ** 2 + yp ** 2)
    zrd = zp
    phi_cp = np.arctan2(yp, xp)
    if phi_cp < 0.0:
        phird = 2.0 * np.pi + phi_cp
    else:
        phird = phi_cp
    return srd, phird, zrd


def azim_factor_bw(phi, fi, isim, ikind):
    fi = np.require(fi, dtype=np.float64)
    factor = lib.azim_factor_bw(
        C.c_double(phi),
        fi.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(isim), C.c_int(ikind))
    return factor


def rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(mt, phi, theta):
    mt = np.require(mt, dtype=np.float64)
    out = np.empty(mt.shape, dtype=np.float64)
    lib.rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(
        mt.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(phi),
        C.c_double(theta),
        out.ctypes.data_as(C.POINTER(C.c_double)))
    return out


def rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(mt, phi, theta):
    np.require(mt, dtype=np.float64)
    out = np.empty(mt.shape, dtype=np.float64)
    lib.rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(
        mt.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(phi),
        C.c_double(theta),
        out.ctypes.data_as(C.POINTER(C.c_double)))
    return out


def rotate_symm_tensor_voigt_xyz_to_src_1d(mt, phi):
    mt = np.require(mt, dtype=np.float64)
    out = np.empty(mt.shape, dtype=np.float64)
    lib.rotate_symm_tensor_voigt_xyz_to_src_1d(
        mt.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(phi),
        out.ctypes.data_as(C.POINTER(C.c_double)))
    return out

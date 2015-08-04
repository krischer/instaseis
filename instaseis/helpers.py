#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Library loading helper.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ctypes as C
import glob
import inspect
import math
import os


LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "lib")

cache = []

WGS_A = 6378137.0
WGS_B = 6356752.314245
FACTOR = 1.0 - (WGS_A - WGS_B) / WGS_A


def load_lib():
    if cache:
        return cache[0]
    else:
        # Enable a couple of different library naming schemes.
        possible_files = glob.glob(os.path.join(LIB_DIR, "instaseis*.so"))
        if not possible_files:
            raise ValueError("Could not find suitable instaseis shared "
                             "library.")
        filename = possible_files[0]
        lib = C.CDLL(filename)
        cache.append(lib)
        return lib


def get_band_code(dt):
    """
    Figure out the channel band code. Done as in SPECFEM.
    """
    if dt <= 0.001:
        band_code = "F"
    elif dt <= 0.004:
        band_code = "C"
    elif dt <= 0.0125:
        band_code = "H"
    elif dt <= 0.1:
        band_code = "B"
    elif dt < 1:
        band_code = "M"
    else:
        band_code = "L"
    return band_code


def wgs84_to_geocentric_latitude(lat):
    """
    Convert a latitude defined on the WGS84 ellipsoid to geocentric
    coordinates.

    Thanks to Kasra Hosseini for the original code snippet!
    """
    # Singularities close to the pole and the equator. Just return the value
    # in that case.
    if abs(lat) < 1E-6 or abs(lat - 90) < 1E-6 or \
            abs(lat + 90.0) < 1E-6:
        return lat

    fac = (WGS_B / WGS_A) ** 2

    colat = math.radians(90.0 - lat)

    geocen_colat = 0.5 * math.pi - math.atan(fac / math.tan(colat))
    return 90.0 - math.degrees(geocen_colat)

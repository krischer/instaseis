#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the source and receiver objects.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

from instaseis import Receiver


def test_station_x_y_z():
    station = Receiver(latitude=42.6390, longitude=74.4940, depth_in_m=0.0)
    assert abs(station.x() - 1252949.21995) < 1E-5
    assert abs(station.y() - 4516152.38916) < 1E-5
    assert abs(station.z() - 4315567.96379) < 1E-5
    assert abs(station.colatitude - 47.3609999) < 1E-5
    assert station.depth_in_m == 0.0
    assert station.radius_in_m() == 6371000.0

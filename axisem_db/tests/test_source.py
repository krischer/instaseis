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

from .. import source


def test_station_x_y_z():
    station = source.SourceOrReceiver(
        latitude=42.6390, longitude=74.4940, depth_in_m=0.0)
    assert abs(station.x - 1252.94921995) < 1E-5
    assert abs(station.y - 4516.15238916) < 1E-5
    assert abs(station.z - 4315.56796379) < 1E-5
    assert abs(station.colatitude - 47.3609999) < 1E-5
    assert station.depth_in_m == 0.0
    assert station.radius_in_m == source.EARTH_RADIUS

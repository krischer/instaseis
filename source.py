#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source and Receiver classes used for the AxiSEM DB Python interface.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np

EARTH_RADIUS = 6371.0 * 1000.0


class SourceOrReceiver(object):
    def __init__(self, latitude, longitude, depth_in_m):
        self.latitude = latitude
        self.longitude = longitude
        self.depth_in_m = depth_in_m

    @property
    def colatitude(self):
        return 90.0 - self.latitude

    @property
    def radius_in_m(self):
        return EARTH_RADIUS - self.depth_in_m

    @property
    def x(self):
        return np.cos(np.deg2rad(self.latitude)) * \
            np.cos(np.deg2rad(self.longitude)) * self.radius_in_m / 1000.0

    @property
    def y(self):
        return np.cos(np.deg2rad(self.latitude)) * \
            np.sin(np.deg2rad(self.longitude)) * self.radius_in_m / 1000.0

    @property
    def z(self):
        return np.sin(np.deg2rad(self.latitude)) * self.radius_in_m / 1000.0


class Source(SourceOrReceiver):
    def __init__(self, latitude, longitude, depth_in_m, m_rr, m_tt, m_pp, m_rt,
                 m_rp, m_tp):
        super(Source, self).__init__(latitude, longitude, depth_in_m)
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp

    @property
    def tensor(self):
        return np.array([self.m_rr, self.m_tt, self.m_pp, self.m_rt, self.m_rp,
                         self.m_tp])

    @property
    def tensor_voigt(self):
        return np.array([self.m_tt, self.m_pp, self.m_rr, self.m_rp, self.m_rt,
                         self.m_tp])


class Receiver(SourceOrReceiver):
    pass


def test_station_x_y_z():
    station = SourceOrReceiver(latitude=42.6390, longitude=74.4940,
                               depth_in_m=0.0)
    assert abs(station.x - 1252.94921995) < 1E-5
    assert abs(station.y - 4516.15238916) < 1E-5
    assert abs(station.z - 4315.56796379) < 1E-5
    assert abs(station.colatitude - 47.3609999) < 1E-5
    assert station.depth_in_m == 0.0
    assert station.radius_in_m == EARTH_RADIUS

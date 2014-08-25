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
from __future__ import absolute_import

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
                 m_rp, m_tp, time_shift=None):
        super(Source, self).__init__(latitude, longitude, depth_in_m)
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp
        self.time_shift = time_shift

    @classmethod
    def from_CMTSOLUTION_file(self, filename):
        f = open(filename, 'r')
        f.readline()
        f.readline()
        time_shift = float(f.readline().split()[2])
        f.readline()
        latitude = float(f.readline().split()[1])
        longitude = float(f.readline().split()[1])
        depth = float(f.readline().split()[1])

        m_rr = float(f.readline().split()[1])
        m_tt = float(f.readline().split()[1])
        m_pp = float(f.readline().split()[1])
        m_rt = float(f.readline().split()[1])
        m_rp = float(f.readline().split()[1])
        m_tp = float(f.readline().split()[1])

        f.close()
        return self(latitude, longitude, depth * 1e3, m_rr, m_tt, m_pp, m_rt,
                    m_rp, m_tp, time_shift)

    @property
    def tensor(self):
        return np.array([self.m_rr, self.m_tt, self.m_pp, self.m_rt, self.m_rp,
                         self.m_tp])

    @property
    def tensor_voigt(self):
        return np.array([self.m_tt, self.m_pp, self.m_rr, self.m_rp, self.m_rt,
                         self.m_tp])


class Receiver(SourceOrReceiver):
    def __init__(self, latitude, longitude):
        super(Receiver, self).__init__(latitude, longitude, depth_in_m=0.0)

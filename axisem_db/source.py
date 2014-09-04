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
from scipy import interp

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
    """
    A class to handle a seimic moment tensor source including a source time
    function.
    """
    def __init__(self, latitude, longitude, depth_in_m, m_rr, m_tt, m_pp, m_rt,
                 m_rp, m_tp, time_shift=None, sliprate=None, dt=None):
        """
        Parameters:
        latitude    -- latitude of the source in degree
        longitude   -- longitude of the source in degree
        depth_in_m  -- source depth in m
        m_rr, m_tt, m_pp, m_rt, m_rp, m_tp
                    -- moment tensor components in r, theta, phi in Nm
        time_shift  -- correction of the origin time in seconds. only useful in
                       the context of finite sources
        sliprate    -- normalized source time function (sliprate)
        dt          -- sampling of the source time function
        """
        super(Source, self).__init__(latitude, longitude, depth_in_m)
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp
        self.time_shift = time_shift

        if sliprate is not None:
            self.sliprate = np.array(sliprate)
        else:
            self.sliprate = None

        self.dt = dt

    @classmethod
    def from_CMTSOLUTION_file(self, filename):
        """
        Initialize a source object from a CMTSOLUTION file.

        parameter:
        filename -- path to the CMTSOLUTION file
        """
        f = open(filename, 'r')
        f.readline()
        f.readline()
        time_shift = float(f.readline().split()[2])
        f.readline()
        latitude = float(f.readline().split()[1])
        longitude = float(f.readline().split()[1])
        depth_in_m = float(f.readline().split()[1]) * 1e3

        m_rr = float(f.readline().split()[1]) / 1e7
        m_tt = float(f.readline().split()[1]) / 1e7
        m_pp = float(f.readline().split()[1]) / 1e7
        m_rt = float(f.readline().split()[1]) / 1e7
        m_rp = float(f.readline().split()[1]) / 1e7
        m_tp = float(f.readline().split()[1]) / 1e7

        f.close()
        return self(latitude, longitude, depth_in_m, m_rr, m_tt, m_pp, m_rt,
                    m_rp, m_tp, time_shift)

    @classmethod
    def from_strike_dip_rake(self, latitude, longitude, depth_in_m, strike,
                             dip, rake, M0, time_shift=None, sliprate=None,
                             dt=None):
        """
        Initialize a source object from a shear source parameterized by strike,
        dip and rake.

        parameter:
        latitude    -- latitude of the source in degree
        longitude   -- longitude of the source in degree
        depth_in_m  -- source depth in m
        strike      -- strike of the fault in degree
        dip         -- dip of the fault in degree
        rake        -- rake of the fault in degree
        M0          -- scalar moment
        time_shift  -- correction of the origin time in seconds. only useful in
                       the context of finite sources
        sliprate    -- normalized source time function (sliprate)
        dt          -- sampling of the source time function
        """
        # formulas in Udias (17.24) are in geographic system North, East,
        # Down, which # transforms to the geocentric as:
        # Mtt =  Mxx, Mpp = Myy, Mrr =  Mzz
        # Mrp = -Myz, Mrt = Mxz, Mtp = -Mxy
        # voigt in tpr: Mtt Mpp Mrr Mrp Mrt Mtp

        phi = np.deg2rad(strike)
        delta = np.deg2rad(dip)
        lambd = np.deg2rad(rake)

        m_tt = (- np.sin(delta) * np.cos(lambd) * np.sin(2. * phi)
                - np.sin(2. * delta) * np.sin(phi)**2. * np.sin(lambd)) * M0

        m_pp = (np.sin(delta) * np.cos(lambd) * np.sin(2. * phi)
                - np.sin(2. * delta) * np.cos(phi)**2. * np.sin(lambd)) * M0

        m_rr = (np.sin(2. * delta) * np.sin(lambd)) * M0

        m_rp = (- np.cos(phi) * np.sin(lambd) * np.cos(2. * delta)
                + np.cos(delta) * np.cos(lambd) * np.sin(phi)) * M0

        m_rt = (- np.sin(lambd) * np.sin(phi) * np.cos(2. * delta)
                - np.cos(delta) * np.cos(lambd) * np.cos(phi)) * M0

        m_tp = (- np.sin(delta) * np.cos(lambd) * np.cos(2. * phi)
                - np.sin(2. * delta) * np.sin(2. * phi) * np.sin(lambd) / 2.) \
            * M0

        return self(latitude, longitude, depth_in_m, m_rr, m_tt, m_pp, m_rt,
                    m_rp, m_tp, time_shift, sliprate, dt)

    @property
    def tensor(self):
        """
        List of moment tensor components in r, theta, phi coordinates:
        [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        """
        return np.array([self.m_rr, self.m_tt, self.m_pp, self.m_rt, self.m_rp,
                         self.m_tp])

    @property
    def tensor_voigt(self):
        """
        List of moment tensor components in theta, phi, r coordinates in Voigt
        notation:
        [m_tt, m_pp, m_rr, m_rp, m_rt, m_tp]
        """
        return np.array([self.m_tt, self.m_pp, self.m_rr, self.m_rp, self.m_rt,
                         self.m_tp])

    def set_sliprate(self, sliprate, dt, normalize=True):
        """
        Add a source time function (sliprate) to a initialized source object.

        Parameters:
        sliprate    -- (normalized) sliprate
        dt          -- sampling of the sliprate
        normalize   -- if sliprate is not normalized, set this to true to
                       normalize it using trapezoidal rule style integration
        """
        self.sliprate = np.array(sliprate)
        if normalize:
            self.sliprate /= np.trapz(sliprate, dx=dt)
        self.dt = dt

    def resample_sliprate(self, dt, nsamp):
        """
        For convolution, the sliprate is needed at the sampling of the fields in
        the database. This function resamples the sliprate using linear
        interpolation.

        Parameters:
        dt      -- desired sampling
        nsamp   -- desired number of samples
        """
        t_new = np.linspace(0, nsamp * dt, nsamp, endpoint=False)
        t_old = np.linspace(0, self.dt * len(self.sliprate),
                            len(self.sliprate), endpoint=False)

        self.sliprate = interp(t_new, t_old, self.sliprate)
        self.dt = dt

    def __str__(self):
        return_str  = 'AxiSEM Database Source:\n'
        return_str += 'longitude : %6.1f s\n' % (self.longitude)
        return_str += 'latitude  : %6.1f s\n' % (self.latitude)
        return_str += 'Mrr       : %10.2e Nm\n' % (self.m_rr)
        return_str += 'Mtt       : %10.2e Nm\n' % (self.m_tt)
        return_str += 'Mpp       : %10.2e Nm\n' % (self.m_pp)
        return_str += 'Mrt       : %10.2e Nm\n' % (self.m_rt)
        return_str += 'Mrp       : %10.2e Nm\n' % (self.m_rp)
        return_str += 'Mtp       : %10.2e Nm\n' % (self.m_tp)

        return return_str


class Receiver(SourceOrReceiver):
    """
    A class to handle a receiver including the name and network.
    """
    def __init__(self, latitude, longitude, name='', network=''):
        """
        latitude    -- latitude of the source in degree
        longitude   -- longitude of the source in degree
        name        -- receiver name
        network     -- network name
        """
        super(Receiver, self).__init__(latitude, longitude, depth_in_m=0.0)
        self.name = name
        self.network = network

    def __str__(self):
        return_str  = 'AxiSEM Database Receiver:\n'
        return_str += 'longitude : %6.1f s\n' % (self.longitude)
        return_str += 'latitude  : %6.1f s\n' % (self.latitude)
        return_str += 'name      : %s\n' % (self.name)
        return_str += 'network   : %s\n' % (self.network)

        return return_str

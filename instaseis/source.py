#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source and Receiver classes of Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import functools
import io
import numpy as np
import obspy
import obspy.core.inventory
from obspy.signal.filter import lowpass
from obspy.signal.util import next_pow_2
import obspy.io.xseed.parser
import os
from scipy import interp

from . import ReceiverParseError, SourceParseError
from . import rotations
from .helpers import (elliptic_to_geocentric_latitude, rfftfreq)

DEFAULT_MU = 32e9


class USGSParamFileParsingException(Exception):
    """
    Custom exception for nice and hopefully save exception passing.
    """
    pass


def _purge_duplicates(f):
    """
    Simple decorator removing duplicates in the returned list. Preserves the
    order and will remove duplicates occuring later in the list.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        ret_val = f(*args, **kwds)
        new_list = []
        for item in ret_val:
            if item in new_list:
                continue
            new_list.append(item)
        return new_list
    return wrapper


def moment2magnitude(M0):
    """
    Convert seismic moment M0 to moment magnitude Mw

    :param M0: seismic moment in Nm
    :type M0: float
    :return Mw: moment magnitude
    :type Mw: float
    """
    return 2.0 / 3.0 * np.log10(M0) - 6.0


def magnitude2moment(Mw):
    """
    Convert moment magnitude Mw to seismic moment M0

    :param Mw: moment magnitude
    :type Mw: float
    :return M0: seismic moment in Nm
    :type M0: float
    """
    return 10.0 ** ((Mw + 6.0) / 2.0 * 3.0)


def fault_vectors_lmn(strike, dip, rake):
    """
    compute vectors l, m = n x l and n describing the fault according to Udias
    Fig. 16.19

    :param strike: strike of the fault measured from North
    :type strike: float
    :param dip: dip of the fault measured from horizontal
    :type dip: float
    :param rake: rake of the fault measured from horizontal
    :type rake: float
    :return (l, m, n): vectors l, m = n x l and n
    :type (l, m, n): tuple of numpy arrays
    """
    phi = np.deg2rad(strike)
    delta = np.deg2rad(dip)
    lambd = np.deg2rad(rake)

    l = np.empty(3)
    m = np.empty(3)
    n = np.empty(3)

    l[0] = np.cos(lambd) * np.cos(phi) \
        + np.cos(delta) * np.sin(lambd) * np.sin(phi)
    l[1] = np.cos(lambd) * np.sin(phi) \
        - np.cos(delta) * np.sin(lambd) * np.cos(phi)
    l[2] = - np.sin(delta) * np.sin(lambd)

    m[0] = - np.sin(lambd) * np.cos(phi) \
        + np.cos(delta) * np.cos(lambd) * np.sin(phi)
    m[1] = - np.sin(lambd) * np.sin(phi) \
        - np.cos(delta) * np.cos(lambd) * np.cos(phi)
    m[2] = - np.sin(delta) * np.cos(lambd)

    n[0] = - np.sin(delta) * np.sin(phi)
    n[1] = np.sin(delta) * np.cos(phi)
    n[2] = - np.cos(delta)

    # Udias 16.99 - 16.104 is in geographic coordinates (North, East, Down),
    # here we use geocentric, i.e. t,p,r

    transform = np.array([-1., 1., -1.])
    l *= transform
    m *= transform
    n *= transform

    return l, m, n


def strike_dip_rake_from_ln(l, n):
    """
    compute strike, dip and rake from the fault vectors l and n
    describing the fault according to Udias Fig. 16.19

    :return (strike, dip, rake): strike, dip and rake
    :type (strike, dip, rake): tuple of floats
    """
    l_norm = l / (l ** 2).sum()
    n_norm = n / (n ** 2).sum()

    delta = np.arccos(n_norm[2])
    phi = np.arctan2(n_norm[0], n_norm[1])

    # needs two different formulas, beqause the first is unstable for dip = 0
    # and the second for dip = 90
    if delta > 0.1:
        lambd = np.arctan2(
            l_norm[2], np.sin(delta) * (-l_norm[0] * np.cos(phi) +
                                        l_norm[1] * np.sin(phi)))
    else:
        lambd = np.arctan2(
            (-l_norm[0] * np.sin(phi) - l_norm[1] * np.cos(phi)),
            np.cos(delta) * (-l_norm[0] * np.cos(phi) +
                             l_norm[1] * np.sin(phi)))

    strike = np.rad2deg(phi)
    dip = np.rad2deg(delta)
    rake = np.rad2deg(lambd)

    return strike, dip, rake


def asymmetric_cosine(trise, tfall=None, npts=10000, dt=0.1):
    """
    Initialize a source time function with asymmetric cosine, normalized to 1

    :param trise: rise time
    :type trise: float
    :param tfall: fall time, defaults to trise
    :type trise: float, optional
    :param npts: number of samples
    :type npts: int, optional
    :param dt: sample interval
    :type dt: float, optional
    """
    # initialize
    if not tfall:
        tfall = trise
    t = np.linspace(0, npts * dt, npts, endpoint=False)
    asc = np.zeros(npts)

    # build slices
    slrise = (t <= trise)
    slfall = np.logical_and(t > trise, t <= trise + tfall)

    # compute stf
    asc[slrise] = (1. - np.cos(np.pi * t[slrise] / trise))
    asc[slfall] = (1. - np.cos(np.pi * (t[slfall] - trise + tfall) / tfall))

    # normalize
    asc /= trise + tfall

    return asc


class SourceOrReceiver(object):
    def __init__(self, latitude, longitude, depth_in_m):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.depth_in_m = float(depth_in_m) if depth_in_m is not None else None

        if not (-90 <= self.latitude <= 90):
            raise ValueError("Invalid latitude value. Latitude must be "
                             "-90 <= x <= 90.")

        if not (-180 <= self.longitude <= 180.0):
            raise ValueError("Invalid longitude value. Longitude must be "
                             "-180 <= x <= 180.")

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def colatitude(self):
        return 90.0 - self.latitude

    @property
    def colatitude_rad(self):
        return np.deg2rad(90.0 - self.latitude)

    @property
    def longitude_rad(self):
        return np.deg2rad(self.longitude)

    @property
    def latitude_rad(self):
        return np.deg2rad(self.latitude)

    def radius_in_m(self, planet_radius=6371e3):
        if self.depth_in_m is None:
            return planet_radius
        else:
            return planet_radius - self.depth_in_m

    def x(self, planet_radius=6371e3):
        return np.cos(np.deg2rad(self.latitude)) * \
            np.cos(np.deg2rad(self.longitude)) * \
            self.radius_in_m(planet_radius=planet_radius)

    def y(self, planet_radius=6371e3):
        return np.cos(np.deg2rad(self.latitude)) * \
            np.sin(np.deg2rad(self.longitude)) * \
            self.radius_in_m(planet_radius=planet_radius)

    def z(self, planet_radius=6371e3):
        return np.sin(np.deg2rad(self.latitude)) * \
            self.radius_in_m(planet_radius=planet_radius)


class Source(SourceOrReceiver):
    """
    Class to handle a seismic moment tensor source including a source time
    function.
    """
    def __init__(self, latitude, longitude, depth_in_m=None, m_rr=0.0,
                 m_tt=0.0, m_pp=0.0, m_rt=0.0, m_rp=0.0, m_tp=0.0,
                 time_shift=None, sliprate=None, dt=None,
                 origin_time=obspy.UTCDateTime(0)):
        """
        :param latitude: geocentric latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param m_rr: moment tensor components in r, theta, phi in Nm
        :param m_tt: moment tensor components in r, theta, phi in Nm
        :param m_pp: moment tensor components in r, theta, phi in Nm
        :param m_rt: moment tensor components in r, theta, phi in Nm
        :param m_rp: moment tensor components in r, theta, phi in Nm
        :param m_tp: moment tensor components in r, theta, phi in Nm
        :param time_shift: correction of the origin time in seconds. Useful
            in the context of finite source or user defined source time
            functions.
        :param sliprate: normalized source time function (sliprate)
        :param dt: sampling of the source time function
        :param origin_time: The origin time of the source. If you don't
            reconvolve with another source time function this time is the
            peak of the source time function used to generate the database.
            If you reconvolve with another source time function this time is
            the time of the first sample of the final seismogram.

        >>> import instaseis
        >>> source = instaseis.Source(
        ...     latitude=89.91, longitude=0.0, depth_in_m=12000,
        ...     m_rr = 4.71e+17, m_tt = 3.81e+15, m_pp =-4.74e+17,
        ...     m_rt = 3.99e+16, m_rp =-8.05e+16, m_tp =-1.23e+17)
        >>> print(source)
        Instaseis Source:
            Longitude        :    0.0 deg
            Latitude         :   89.9 deg
            Depth            : 1.2e+01 km
            Moment Magnitude :   5.80
            Scalar Moment    :   4.96e+17 Nm
            Mrr              :   4.71e+17 Nm
            Mtt              :   3.81e+15 Nm
            Mpp              :  -4.74e+17 Nm
            Mrt              :   3.99e+16 Nm
            Mrp              :  -8.05e+16 Nm
            Mtp              :  -1.23e+17 Nm
        """
        super(Source, self).__init__(latitude, longitude, depth_in_m)
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp
        self.origin_time = origin_time
        self.time_shift = time_shift
        self.sliprate = np.array(sliprate) if sliprate is not None else None
        self.dt = dt

    @staticmethod
    def parse(filename_or_obj):
        """
        Attempts to parse anything to a Source object. Can currently read
        anything ObsPy can read, ObsPy event related objects.

        For anything ObsPy related, it must contain a full moment tensor,
        otherwise it will raise an error.

        Coordinates are assumed to be defined on the WGS84 ellipsoid and
        will be converted to geocentric coordinates.

        :param filename_or_obj: The object or filename to parse.


        The following example will read a local QuakeML file and return a
        :class:`~instaseis.source.Source` object.

        >>> import instaseis
        >>> source = instaseis.Source.parse("quake.xml")
        >>> print(source)
        Instaseis Source:
            Longitude        :   -3.5 deg
            Latitude         :   37.0 deg
            Depth            : 6.1e+02 km
            Moment Magnitude :   6.41
            Scalar Moment    :   4.16e+18 Nm
            Mrr              :  -2.16e+18 Nm
            Mtt              :   5.36e+17 Nm
            Mpp              :   1.62e+18 Nm
            Mrt              :   1.30e+16 Nm
            Mrp              :   3.23e+18 Nm
            Mtp              :   1.75e+18 Nm
        """
        if isinstance(filename_or_obj, (str, bytes)):
            # Anything ObsPy can read.
            try:
                src = obspy.read_events(filename_or_obj)
            except:
                pass
            else:
                return Source.parse(src)
            raise SourceParseError("Could not parse the given source.")
        elif isinstance(filename_or_obj, obspy.Catalog):
            if len(filename_or_obj) == 0:
                raise SourceParseError("Event catalog contains zero events.")
            elif len(filename_or_obj) > 1:
                raise SourceParseError(
                    "Event catalog contains %i events. Only one is allowed. "
                    "Please parse seperately." % len(filename_or_obj))
            return Source.parse(filename_or_obj[0])
        elif isinstance(filename_or_obj, obspy.core.event.Event):
            ev = filename_or_obj
            if not ev.origins:
                raise SourceParseError("Event must contain an origin.")
            if not ev.focal_mechanisms:
                raise SourceParseError("Event must contain a focal mechanism.")
            org = ev.preferred_origin() or ev.origins[0]
            fm = ev.preferred_focal_mechanism() or ev.focal_mechanisms[0]
            if not fm.moment_tensor:
                raise SourceParseError("Event must contain a moment tensor.")
            t = fm.moment_tensor.tensor
            return Source(
                latitude=elliptic_to_geocentric_latitude(org.latitude),
                longitude=org.longitude,
                depth_in_m=org.depth,
                origin_time=org.time,
                m_rr=t.m_rr,
                m_tt=t.m_tt,
                m_pp=t.m_pp,
                m_rt=t.m_rt,
                m_rp=t.m_rp,
                m_tp=t.m_tp)
        else:
            raise NotImplementedError

    @classmethod
    def from_strike_dip_rake(self, latitude, longitude, depth_in_m, strike,
                             dip, rake, M0, time_shift=None, sliprate=None,
                             dt=None, origin_time=obspy.UTCDateTime(0)):
        """
        Initialize a source object from a shear source parameterized by strike,
        dip and rake.

        :param latitude: geocentric latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param strike: strike of the fault in degree
        :param dip: dip of the fault in degree
        :param rake: rake of the fault in degree
        :param M0: scalar moment
        :param time_shift: correction of the origin time in seconds. only
            useful in the context of finite sources
        :param sliprate: normalized source time function (sliprate)
        :param dt: sampling of the source time function
        :param origin_time: The origin time of the source. If you don't
            reconvolve with another source time function this time is the
            peak of the source time function used to generate the database.
            If you reconvolve with another source time function this time is
            the time of the first sample of the final seismogram.

        >>> import instaseis
        >>> source = instaseis.Source.from_strike_dip_rake(
        ...     latitude=10.0, longitude=12.0, depth_in_m=1000, strike=79,
        ...     dip=10, rake=20, M0=1E17)
        >>> print(source)
        Instaseis Source:
            Longitude        :   12.0 deg
            Latitude         :   10.0 deg
            Depth            : 1.0e+00 km
            Moment Magnitude :   5.33
            Scalar Moment    :   1.00e+17 Nm
            Mrr              :   1.17e+16 Nm
            Mtt              :  -1.74e+16 Nm
            Mpp              :   5.69e+15 Nm
            Mrt              :  -4.92e+16 Nm
            Mrp              :   8.47e+16 Nm
            Mtp              :   1.29e+16 Nm
        """
        assert M0 >= 0
        if dt is not None:
            assert dt > 0

        # formulas in Udias (17.24) are in geographic system North, East,
        # Down, which # transforms to the geocentric as:
        # Mtt =  Mxx, Mpp = Myy, Mrr =  Mzz
        # Mrp = -Myz, Mrt = Mxz, Mtp = -Mxy
        # voigt in tpr: Mtt Mpp Mrr Mrp Mrt Mtp
        phi = np.deg2rad(strike)
        delta = np.deg2rad(dip)
        lambd = np.deg2rad(rake)

        m_tt = (- np.sin(delta) * np.cos(lambd) * np.sin(2. * phi) -
                np.sin(2. * delta) * np.sin(phi)**2. * np.sin(lambd)) * M0

        m_pp = (np.sin(delta) * np.cos(lambd) * np.sin(2. * phi) -
                np.sin(2. * delta) * np.cos(phi)**2. * np.sin(lambd)) * M0

        m_rr = (np.sin(2. * delta) * np.sin(lambd)) * M0

        m_rp = (- np.cos(phi) * np.sin(lambd) * np.cos(2. * delta) +
                np.cos(delta) * np.cos(lambd) * np.sin(phi)) * M0

        m_rt = (- np.sin(lambd) * np.sin(phi) * np.cos(2. * delta) -
                np.cos(delta) * np.cos(lambd) * np.cos(phi)) * M0

        m_tp = (- np.sin(delta) * np.cos(lambd) * np.cos(2. * phi) -
                np.sin(2. * delta) * np.sin(2. * phi) * np.sin(lambd) / 2.) * \
            M0

        source = self(latitude, longitude, depth_in_m, m_rr, m_tt, m_pp, m_rt,
                      m_rp, m_tp, time_shift, sliprate, dt,
                      origin_time=origin_time)

        # storing strike, dip and rake for plotting purposes
        source.phi = phi
        source.delta = delta
        source.lambd = lambd

        return source

    @property
    def M0(self):
        """
        Scalar Moment M0 in Nm
        """
        return (self.m_rr ** 2 + self.m_tt ** 2 + self.m_pp ** 2 +
                2 * self.m_rt ** 2 + 2 * self.m_rp ** 2 +
                2 * self.m_tp ** 2) ** 0.5 * 0.5 ** 0.5

    @property
    def moment_magnitude(self):
        """
        Moment magnitude M_w
        """
        return moment2magnitude(self.M0)

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

    def set_sliprate(self, sliprate, dt, time_shift=None, normalize=True):
        """
        Add a source time function (sliprate) to a initialized source object.

        :param sliprate: (normalized) sliprate
        :param dt: sampling of the sliprate
        :param normalize: if sliprate is not normalized, set this to true to
            normalize it using trapezoidal rule style integration
        """
        self.sliprate = np.array(sliprate)
        if normalize:
            self.sliprate /= np.trapz(sliprate, dx=dt)
        self.dt = dt
        self.time_shift = time_shift

    def resample_sliprate(self, dt, nsamp):
        """
        For convolution, the sliprate is needed at the sampling of the fields
        in the database. This function resamples the sliprate using linear
        interpolation.

        :param dt: desired sampling
        :param nsamp: desired number of samples
        """
        t_new = np.linspace(0, nsamp * dt, nsamp, endpoint=False)
        t_old = np.linspace(0, self.dt * len(self.sliprate),
                            len(self.sliprate), endpoint=False)

        self.sliprate = interp(t_new, t_old, self.sliprate)
        self.dt = dt

    def set_sliprate_dirac(self, dt, nsamp):
        """
        :param dt: desired sampling
        :param nsamp: desired number of samples
        """
        self.sliprate = np.zeros(nsamp)
        self.sliprate[0] = 1.0 / dt
        self.dt = dt

    def set_sliprate_lp(self, dt, nsamp, freq, corners=4, zerophase=False):
        """
        :param dt: desired sampling
        :param nsamp: desired number of samples
        """
        self.sliprate = np.zeros(nsamp)
        self.sliprate[0] = 1.0 / dt
        self.sliprate = lowpass(self.sliprate, freq, 1./dt, corners, zerophase)
        self.dt = dt

    def normalize_sliprate(self):
        """
        normalize the sliprate using trapezoidal rule
        """
        self.sliprate /= np.trapz(self.sliprate, dx=self.dt)

    def lp_sliprate(self, freq, corners=4, zerophase=False):
        self.sliprate = lowpass(self.sliprate, freq, 1./self.dt, corners,
                                zerophase)

    def __str__(self):
        return_str = 'Instaseis Source:\n'
        return_str += '\torigin time      : %s\n' % (self.origin_time,)
        return_str += '\tLongitude        : %6.1f deg\n' % (self.longitude,)
        return_str += '\tLatitude         : %6.1f deg\n' % (self.latitude,)
        return_str += '\tDepth            : %s km\n' % (
            "%6.1e km" % (self.depth_in_m / 1e3)
            if self.depth_in_m is not None else " not set")
        return_str += '\tMoment Magnitude :   %4.2f\n' \
                      % (self.moment_magnitude,)
        return_str += '\tScalar Moment    : %10.2e Nm\n' % (self.M0,)
        return_str += '\tMrr              : %10.2e Nm\n' % (self.m_rr,)
        return_str += '\tMtt              : %10.2e Nm\n' % (self.m_tt,)
        return_str += '\tMpp              : %10.2e Nm\n' % (self.m_pp,)
        return_str += '\tMrt              : %10.2e Nm\n' % (self.m_rt,)
        return_str += '\tMrp              : %10.2e Nm\n' % (self.m_rp,)
        return_str += '\tMtp              : %10.2e Nm\n' % (self.m_tp,)

        return return_str


class ForceSource(SourceOrReceiver):
    """
    Class to handle a seismic force source.

    :param latitude: geocentric latitude of the source in degree
    :param longitude: longitude of the source in degree
    :param depth_in_m: source depth in m
    :param f_r: force components in r, theta, phi in N
    :param f_t: force components in r, theta, phi in N
    :param f_p: force components in r, theta, phi in N
    :param origin_time: The origin time of the source. If you don't
        reconvolve with another source time function this time is the
        peak of the source time function used to generate the database.
        If you reconvolve with another source time function this time is
        the time of the first sample of the final seismogram.
    :param sliprate: normalized source time function (sliprate)


    >>> import instaseis
    >>> source = instaseis.ForceSource(latitude=12.34, longitude=12.34,
    ...                                f_r=1E10)
    >>> print(source)
    Instaseis Force Source:
        longitude :   12.3 deg
        latitude  :   12.3 deg
        Fr        :   1.00e+10 N
        Ft        :   0.00e+00 N
        Fp        :   0.00e+00 N
    """
    def __init__(self, latitude, longitude, depth_in_m=None, f_r=0., f_t=0.,
                 f_p=0., origin_time=obspy.UTCDateTime(0), sliprate=None):
        super(ForceSource, self).__init__(latitude, longitude, depth_in_m)
        self.f_r = f_r
        self.f_t = f_t
        self.f_p = f_p
        self.origin_time = origin_time
        self.sliprate = sliprate

    @property
    def force_tpr(self):
        """
        List of force components in theta, phi, r coordinates:
        [f_t, f_p, f_r]
        """
        return np.array([self.f_t, self.f_p, self.f_r])

    @property
    def force_rtp(self):
        """
        List of force components in r, theta, phi, coordinates:
        [f_r, f_t, f_p]
        """
        return np.array([self.f_r, self.f_t, self.f_p])

    def __str__(self):
        return_str = 'Instaseis Force Source:\n'
        return_str += '\torigin time      : %s\n' % (self.origin_time,)
        return_str += '\tlongitude : %6.1f deg\n' % (self.longitude)
        return_str += '\tlatitude  : %6.1f deg\n' % (self.latitude)
        return_str += '\tFr        : %10.2e N\n' % (self.f_r)
        return_str += '\tFt        : %10.2e N\n' % (self.f_t)
        return_str += '\tFp        : %10.2e N\n' % (self.f_p)

        return return_str


class Receiver(SourceOrReceiver):
    """
    Class dealing with seismic receivers.

    :type latitude: float
    :param latitude: The geocentric latitude of the receiver in degree.
    :type longitude: float
    :param longitude: The longitude of the receiver in degree.
    :type depth_in_m: float
    :param depth_in_m: The depth of the receiver in meters. Only
    :type network: str, optional
    :param network: The network id of the receiver.
    :type station: str, optional
    :param station: The station id of the receiver.
    :type location: str
    :param location: The location code of the receiver.

    >>> from instaseis import Receiver
    >>> rec = Receiver(latitude=12.34, longitude=56.78, network="AB",
    ...                station="CDE", location="SY")
    >>> print(rec)
    Instaseis Receiver:
    longitude :   56.8 deg
    latitude  :   12.3 deg
    network   : AB
    station   : CDE
    location  : SY
    """
    def __init__(self, latitude, longitude, network=None, station=None,
                 location=None, depth_in_m=None):
        super(Receiver, self).__init__(latitude, longitude,
                                       depth_in_m=depth_in_m)
        self.network = network or ""
        self.network = self.network.strip()
        assert len(self.network) <= 2

        self.station = station or ""
        self.station = self.station.strip()
        assert len(self.station) <= 5

        self.location = location or ""
        self.location = self.location.strip()
        assert len(self.location) <= 2

    def __str__(self):
        return_str = 'Instaseis Receiver:\n'
        return_str += '\tlongitude : %6.1f deg\n' % (self.longitude)
        return_str += '\tlatitude  : %6.1f deg\n' % (self.latitude)
        return_str += '\tnetwork   : %s\n' % (self.network)
        return_str += '\tstation   : %s\n' % (self.station)
        return_str += '\tlocation  : %s\n' % (self.location)

        return return_str

    @staticmethod
    @_purge_duplicates
    def parse(filename_or_obj, network_code=None):
        """
        Attempts to parse anything to a list of
        :class:`~instaseis.source.Receiver` objects. Always
        returns a list, even if it only contains a single element. It is
        meant as a single entry point for receiver information from any source.

        Supports StationXML, the custom STATIONS fileformat, SAC files,
        SEED files, and a number of ObsPy objects. This method can
        furthermore work with anything ObsPy can deal with (filename, URL,
        memory files, ...).

        Coordinates are assumed to be defined on the WGS84 ellipsoid and
        will be converted to geocentric coordinates.

        :param filename_or_obj: Filename/URL/Python object
        :param network_code: Network code needed to parse ObsPy station
            objects. Likely only needed for the recursive part of this method.
        :return: List of :class:`~instaseis.source.Receiver` objects.

        The following example parses a StationXML file to a list of
        :class:`~instaseis.source.Receiver` objects.

        >>> import instaseis
        >>> print(instaseis.Receiver.parse("TA.Q56A..BH.xml"))
        [<instaseis.source.Receiver object at 0x...>]
        """
        receivers = []

        # STATIONS file.
        if isinstance(filename_or_obj, (str, bytes)) and \
                os.path.exists(filename_or_obj):
            try:
                return Receiver._parse_stations_file(filename_or_obj)
            except:
                pass
        # ObsPy inventory.
        elif isinstance(filename_or_obj, obspy.core.inventory.Inventory):
            for network in filename_or_obj:
                receivers.extend(Receiver.parse(network))
            return receivers
        # ObsPy network.
        elif isinstance(filename_or_obj, obspy.core.inventory.Network):
            for station in filename_or_obj:
                receivers.extend(Receiver.parse(
                    station, network_code=filename_or_obj.code))
            return receivers
        # ObsPy station.
        elif isinstance(filename_or_obj, obspy.core.inventory.Station):
            # If there are no channels, use the station coordinates.
            if not filename_or_obj.channels:
                return [Receiver(
                    latitude=elliptic_to_geocentric_latitude(
                        filename_or_obj.latitude),
                    longitude=filename_or_obj.longitude,
                    network=network_code, station=filename_or_obj.code)]
            # Otherwise use the channel information. Raise an error if the
            # coordinates are not identical for each channel. Only parse
            # latitude and longitude, as the DB currently cannot deal with
            # varying receiver heights.
            else:
                coords = set((_i.latitude, _i.longitude) for _i in
                             filename_or_obj.channels)
                if len(coords) != 1:
                    raise ReceiverParseError(
                        "The coordinates of the channels of station '%s.%s' "
                        "are not identical." % (network_code,
                                                filename_or_obj.code))
                coords = coords.pop()
                return [Receiver(
                    latitude=elliptic_to_geocentric_latitude(coords[0]),
                    longitude=coords[1],
                    network=network_code,
                    station=filename_or_obj.code)]
        # ObsPy Stream (SAC files contain coordinates).
        elif isinstance(filename_or_obj, obspy.Stream):
            for tr in filename_or_obj:
                receivers.extend(Receiver.parse(tr))
            return receivers
        elif isinstance(filename_or_obj, obspy.Trace):
            if not hasattr(filename_or_obj.stats, "sac"):
                raise ReceiverParseError("ObsPy Trace must have an sac "
                                         "attribute.")
            if "stla" not in filename_or_obj.stats.sac or \
                    "stlo" not in filename_or_obj.stats.sac:
                raise ReceiverParseError(
                    "SAC file does not contain coordinates for channel '%s'" %
                    filename_or_obj.id)
            coords = (filename_or_obj.stats.sac.stla,
                      filename_or_obj.stats.sac.stlo)
            return [Receiver(
                latitude=elliptic_to_geocentric_latitude(coords[0]),
                longitude=coords[1],
                network=filename_or_obj.stats.network,
                station=filename_or_obj.stats.station)]
        elif isinstance(filename_or_obj, obspy.io.xseed.parser.Parser):
            inv = filename_or_obj.get_inventory()
            stations = collections.defaultdict(list)
            for chan in inv["channels"]:
                stat = tuple(chan["channel_id"].split(".")[:2])
                stations[stat].append((chan["latitude"], chan["longitude"]))
            receivers = []
            for key, value in stations.items():
                if len(set(value)) != 1:
                    raise ReceiverParseError(
                        "The coordinates of the channels of station '%s.%s' "
                        "are not identical" % key)
                receivers.append(Receiver(
                    latitude=elliptic_to_geocentric_latitude(value[0][0]),
                    longitude=value[0][1],
                    network=key[0],
                    station=key[1]))
            return receivers

        # Check if its anything ObsPy can read and recurse.
        try:
            return Receiver.parse(obspy.read_inventory(filename_or_obj))
        except ReceiverParseError as e:
            raise e
        except:
            pass

        # SAC files contain station coordinates.
        try:
            return Receiver.parse(obspy.read(filename_or_obj))
        except ReceiverParseError as e:
            raise e
        except:
            pass

        # Last but not least try to parse it as a SEED file.
        try:
            return Receiver.parse(
                obspy.io.xseed.parser.Parser(filename_or_obj))
        except ReceiverParseError as e:
            raise e
        except:
            pass

        raise ValueError("%s could not be parsed." % repr(filename_or_obj))

    @staticmethod
    def _parse_stations_file(filename):
        """
        Parses a custom STATIONS file format to a list of Receiver objects.

        Coordinates are assumed to be defined on the WGS84 ellipsoid and
        will be converted to geocentric coordinates.

        :param filename: Filename
        :return: List of :class:`~instaseis.source.Receiver` objects.
        """
        with open(filename, 'rt') as f:
            receivers = []

            for line in f:
                station, network, lat, lon, _, _ = line.split()
                lat = elliptic_to_geocentric_latitude(float(lat))
                lon = float(lon)
                receivers.append(Receiver(lat, lon, network, station))

        return receivers


class FiniteSource(object):
    """
    A class to handle finite sources represented by a number of point sources.

    :param pointsources: The points sources making up the finite source.
    :type pointsources: list of :class:`~instaseis.source.Source` objects
    :param CMT: The centroid of the finite source.
    :type CMT: :class:`~instaseis.source.Source`, optional
    :param magnitude: The total moment magnitude of the source.
    :type magnitude: float, optional
    :param event_duration: The event duration in seconds.
    :type event_duration: float, optional
    :param hypocenter_longitude: The hypocentral longitude.
    :type hypocenter_longitude: float, optional
    :param hypocenter_latitude: The hypocentral latitude.
    :type hypocenter_latitude: float, optional
    :param hypocenter_depth_in_m: The hypocentral depth in m.
    :type hypocenter_depth_in_m: float, optional
    """
    def __init__(self, pointsources=None, CMT=None, magnitude=None,
                 event_duration=None, hypocenter_longitude=None,
                 hypocenter_latitude=None, hypocenter_depth_in_m=None):
        self.pointsources = pointsources
        self.CMT = CMT
        self.magnitude = magnitude
        self.event_duration = event_duration
        self.hypocenter_longitude = hypocenter_longitude
        self.hypocenter_latitude = hypocenter_latitude
        self.hypocenter_depth_in_m = hypocenter_depth_in_m
        self.current = 0

    def __len__(self):
        return len(self.pointsources)

    def __iter__(self):
        return self

    def next(self):  # pragma: no cover
        """
        For Py2K compat.
        """
        return self.__next__()

    def __next__(self):
        if self.pointsources is None:
            raise ValueError('FiniteSource not Initialized')
        if self.current > len(self.pointsources) - 1:
            self.current = 0
            raise StopIteration
        else:
            self.current += 1
            return self.pointsources[self.current - 1]

    def __getitem__(self, index):
        return self.pointsources[index]

    @classmethod
    def from_srf_file(self, filename, normalize=False):
        """
        Initialize a finite source object from a 'standard rupture format'
        (.srf) file

        Coordinates are assumed to be defined on the WGS84 ellipsoid and
        will be converted to geocentric coordinates.

        :param filename: path to the .srf file
        :type filename: str
        :param normalize: normalize the sliprate to 1
        :type normalize: bool, optional

        >>> import instaseis
        >>> source = instaseis.FiniteSource.from_srf_file("filename.srf")
        >>> print(source)
        Instaseis Finite Source:
            Moment Magnitude     : 7.09
            scalar Moment        : 6.67e+20 Nm
            #point sources       : 117414
            rupture duration     :  131.5 s
            time shift           :    0.7 s
            min depth            :   24.3 m
            max depth            : 18170.8 m
            hypocenter depth     : 18170.8 m
            min latitude         :   23.3 deg
            max latitude         :   24.7 deg
            hypocenter latitude  :   23.3 deg
            min longitude        : -148.5 deg
            max longitude        : -145.7 deg
            hypocenter longitude : -145.7 deg
        """
        with open(filename, "rt") as f:
            # go to POINTS block
            line = f.readline()
            while 'POINTS' not in line:
                line = f.readline()

            npoints = int(line.split()[1])
            sources = []

            for _ in np.arange(npoints):
                lon, lat, dep, stk, dip, area, tinit, dt = \
                    map(float, f.readline().split())

                # Convert latitude to a geocentric latitude.
                lat = elliptic_to_geocentric_latitude(lat)

                rake, slip1, nt1, slip2, nt2, slip3, nt3 = \
                    map(float, f.readline().split())

                dep *= 1e3     # km   > m
                area *= 1e-4   # cm^2 > m^2
                slip1 *= 1e-2  # cm   > m
                slip2 *= 1e-2  # cm   > m
                # slip3 *= 1e-2  # cm   > m

                nt1, nt2, nt3 = map(int, (nt1, nt2, nt3))

                if nt1 > 0:
                    line = f.readline()
                    while len(line.split()) < nt1:
                        line = line + f.readline()
                    stf = np.array(line.split(), dtype=float)
                    if normalize:
                        stf /= np.trapz(stf, dx=dt)

                    M0 = area * DEFAULT_MU * slip1

                    sources.append(
                        Source.from_strike_dip_rake(
                            lat, lon, dep, stk, dip, rake, M0,
                            time_shift=tinit, sliprate=stf, dt=dt))

                if nt2 > 0:
                    line = f.readline()
                    while len(line.split()) < nt2:
                        line = line + f.readline()
                    stf = np.array(line.split(), dtype=float)
                    if normalize:
                        stf /= np.trapz(stf, dx=dt)

                    M0 = area * DEFAULT_MU * slip2

                    sources.append(
                        Source.from_strike_dip_rake(
                            lat, lon, dep, stk, dip, rake, M0,
                            time_shift=tinit, sliprate=stf, dt=dt))

                if nt3 > 0:
                    raise NotImplementedError('Slip along u3 axis')

            return self(pointsources=sources)

    @classmethod
    def from_usgs_param_file(cls, filename_or_obj, npts=10000, dt=0.1,
                             trise_min=1.0):
        """
        Initialize a finite source object from a (.param) file available from
        the USGS website

        Coordinates are assumed to be defined on the WGS84 ellipsoid and
        will be converted to geocentric coordinates.

        :param filename_or_obj: path to the .param file
        :type filename_or_obj: str
        :param npts: number of samples for the source time functions. Should be
            enough to accommodate long rise times plus optional filter
            responses.
        :type npts: int
        :param dt: sample interval for the source time functions. Needs to be
            small enough to sample short rise times and can then be low pass
            filtered and downsampled before extracting seismograms.
        :type dt: float
        :param trise_min: minimum rise time. If rise time or fall time is
            smaller than this value, it is replaced by the minimum value.
            Mostly meant to handle zero rise times in some USGS files.
        :type trise_min: float

        >>> import instaseis
        >>> source = instaseis.FiniteSource.from_srf_file("filename.param")
        >>> print(source)
        Instaseis Finite Source:
            Moment Magnitude     : 7.09
            scalar Moment        : 6.67e+20 Nm
            #point sources       : 117414
            rupture duration     :  131.5 s
            time shift           :    0.7 s
            min depth            :   24.3 m
            max depth            : 18170.8 m
            hypocenter depth     : 18170.8 m
            min latitude         :   23.3 deg
            max latitude         :   24.7 deg
            hypocenter latitude  :   23.3 deg
            min longitude        : -148.5 deg
            max longitude        : -145.7 deg
            hypocenter longitude : -145.7 deg
        """
        if hasattr(filename_or_obj, "readline"):
            return cls._from_usgs_param_file(
                fh=filename_or_obj, npts=npts, dt=dt, trise_min=trise_min)

        with io.open(filename_or_obj, "rb") as fh:
            return cls._from_usgs_param_file(fh=fh, npts=npts, dt=dt,
                                             trise_min=trise_min)

    @classmethod
    def _from_usgs_param_file(cls, fh, npts, dt, trise_min):
        """
        Internal function actually reading a USGS param file from any open
        binary buffer.
        """
        # number of segments
        line = fh.readline().decode().strip()
        if not line.startswith("#Total number of fault_segments"):
            raise USGSParamFileParsingException("Not a valid USGS param file.")
        nseg = int(line.split()[-1])
        sources = []

        # parse all segments
        for _ in range(nseg):

            # got to point source segment
            for line in fh:
                line = line.decode()
                if '#Lat. Lon. depth' in line:
                    break

            # read all point sources until reaching next segment
            for line in fh:
                line = line.decode()
                if '#Fault_segment' in line:
                    break

                # Lat. Lon. depth slip rake strike dip t_rup t_ris t_fal mo
                (lat, lon, dep, slip, rake, stk, dip, tinit, trise, tfall,
                    M0) = map(float, line.split())

                # Negative rupture times are not supported with the current
                # logic.
                if tinit < 0:  # pragma: no cover
                    raise USGSParamFileParsingException(
                        "File contains a negative rupture time "
                        "which Instaseis cannot currently deal "
                        "with.")

                # Calculate the end time.
                endtime = trise + tfall
                if endtime > (npts - 1) * dt:
                    raise USGSParamFileParsingException(
                        "Rise + fall time are longer than the "
                        "total length of calculated slip. "
                        "Please use more samples.")

                # Convert latitude to a geocentric latitude.
                lat = elliptic_to_geocentric_latitude(lat)

                dep *= 1e3    # km > m
                slip *= 1e-2  # cm > m
                M0 *= 1e-7    # dyn / cm > N * m

                # These checks also take care of negative times.
                if trise < trise_min:
                    trise = trise_min

                if tfall < trise_min:
                    tfall = trise_min

                stf = asymmetric_cosine(trise, tfall, npts, dt)
                sources.append(
                    Source.from_strike_dip_rake(
                        lat, lon, dep, stk, dip, rake, M0,
                        time_shift=tinit, sliprate=stf, dt=dt))

        if not sources:
            raise USGSParamFileParsingException(
                "No point sources found in the file.")

        return cls(pointsources=sources)

    @classmethod
    def from_Haskell(self, latitude, longitude, depth_in_m, strike, dip, rake,
                     M0, fault_length, fault_width, rupture_velocity, nl=100,
                     nw=1, trise=1., tfall=None, dt=0.1, planet_radius=6371e3,
                     origin_time=obspy.UTCDateTime(0)):
        """
        Initialize a source object from a shear source parameterized by strike,
        dip and rake.

        :param latitude: geocentric latitude of the source centroid in degree
        :param longitude: longitude of the source centroid in degree
        :param depth_in_m: source centroid depth in m
        :param strike: strike of the fault in degree
        :param dip: dip of the fault in degree
        :param rake: rake of the fault in degree
        :param M0: scalar moment
        :param fault_length: fault length in m
        :param fault_width: fault width in m
        :param rupture_velocity: rupture velocity in m / s. Use negative value
            to propagate the rupture in negative rake direction.
        :param nl: number of point sources along strike direction
        :param nw: number of point sources perpendicular strike direction
        :param trise: rise time
        :param tfall: fall time
        :param dt: sampling of the source time function
        :param planet_radius: radius of the planet, default to Earth.
        :param origin_time: The origin time of the first patch breaking.
        """
        # raise NotImplementedError

        sources = []
        nsources = nl * nw

        colatitude = 90. - latitude
        longitude_rad = np.radians(longitude)
        # latitude_rad = np.radians(latitude)
        colatitude_rad = np.radians(colatitude)

        # centroid in global cartesian coordinates
        centroid_xyz = rotations.coord_transform_lat_lon_depth_to_xyz(
            latitude, longitude, depth_in_m, planet_radius)

        # compute fault vectors and transform to global cartesian system
        l_src, m_src, n_src = fault_vectors_lmn(strike, dip, rake)
        l_xyz = rotations.rotate_vector_xyz_src_to_xyz_earth(
            l_src, longitude_rad, colatitude_rad)
        m_xyz = rotations.rotate_vector_xyz_src_to_xyz_earth(
            m_src, longitude_rad, colatitude_rad)
        n_xyz = rotations.rotate_vector_xyz_src_to_xyz_earth(
            n_src, longitude_rad, colatitude_rad)

        # make a mesh centered on patch centers, xi1 and xi2 as defined by Aki
        # and Richards, Fig 10.2
        xi1, step = np.linspace(-.5 * fault_length, .5 * fault_length, nl,
                                endpoint=False, retstep=True)
        xi1 = xi1 + step / 2.

        xi2, step = np.linspace(-.5 * fault_width, .5 * fault_width, nw,
                                endpoint=False, retstep=True)
        xi2 = xi2 + step / 2.

        xi1_mesh, xi2_mesh = np.meshgrid(xi1, xi2)
        xi1_mesh = xi1_mesh.flatten()
        xi2_mesh = xi2_mesh.flatten()

        # create point sources in cartesian coordinates
        src_xyz = centroid_xyz.repeat(nsources).reshape((3, nsources)) \
            + np.outer(l_xyz, xi1_mesh) + np.outer(m_xyz, xi2_mesh)

        # transform to lat, lon, depth
        src_lat, src_lon, src_depth = \
            rotations.coord_transform_xyz_to_lat_lon_depth(
                src_xyz[0, :], src_xyz[1, :], src_xyz[2, :],
                planet_radius=planet_radius)
        src_colat = 90. - src_lat

        # make sure all points are inside the planet
        if np.any(src_depth < 0):
            raise ValueError('At least one source point outside planet, '
                             'maximum height in m: %f' % (-src_depth.min(),))

        # compute time shifts as distance along xi1
        time_shift = xi1_mesh / rupture_velocity
        # time shifts should be positive
        time_shift -= time_shift.min()

        # generate source time function (same for all point sources)
        if tfall:
            npts = int((trise + tfall) / dt) + 1
        else:
            npts = int((trise * 2) / dt) + 1
        stf = asymmetric_cosine(trise, tfall, npts, dt)

        for i in np.arange(nsources):
            # compute strike dip and rake in the coordinate system of each
            # source point
            l_src = rotations.rotate_vector_xyz_earth_to_xyz_src(
                l_xyz, np.deg2rad(src_lon[i]), np.deg2rad(src_colat[i]))
            n_src = rotations.rotate_vector_xyz_earth_to_xyz_src(
                n_xyz, np.deg2rad(src_lon[i]), np.deg2rad(src_colat[i]))
            strik, dip, rake = strike_dip_rake_from_ln(l_src, n_src)

            # initialize point source
            src = Source.from_strike_dip_rake(
                src_lat[i], src_lon[i], src_depth[i], strike, dip, rake,
                M0 / nsources, time_shift=time_shift[i],
                origin_time=origin_time, dt=dt, sliprate=stf)
            sources.append(src)

        # return as FiniteSource
        return self(pointsources=sources)

    def resample_sliprate(self, dt, nsamp):
        """
        For convolution, the sliprate is needed at the sampling of the fields
        in the database. This function resamples the sliprate using linear
        interpolation for all pointsources in the finite source.

        :param dt: desired sampling
        :param nsamp: desired number of samples
        """
        for ps in self.pointsources:
            ps.resample_sliprate(dt, nsamp)

    def set_sliprate_dirac(self, dt, nsamp):
        """
        :param dt: desired sampling
        :param nsamp: desired number of samples
        """
        for ps in self.pointsources:
            ps.set_sliprate_dirac(dt, nsamp)

    def set_sliprate_lp(self, dt, nsamp, freq, corners=4, zerophase=False):
        """
        :param dt: desired sampling
        :param nsamp: desired number of samples
        """
        for ps in self.pointsources:
            ps.set_sliprate_lp(dt, nsamp, freq, corners, zerophase)

    def normalize_sliprate(self):
        """
        normalize the sliprate using trapezoidal rule
        """
        for ps in self.pointsources:
            ps.normalize_sliprate()

    def lp_sliprate(self, freq, corners=4, zerophase=False):
        for ps in self.pointsources:
            ps.lp_sliprate(freq, corners, zerophase)

    def find_hypocenter(self):
        """
        Finds the hypo- and epicenter based on the point source that has the
        smallest timeshift
        """
        ps_hypo = min(self.pointsources, key=lambda x: x.time_shift)
        self.hypocenter_longitude = ps_hypo.longitude
        self.hypocenter_latitude = ps_hypo.latitude
        self.hypocenter_depth_in_m = ps_hypo.depth_in_m

    def compute_centroid(self, planet_radius=6371e3, dt=None, nsamp=None):
        """
        computes the centroid moment tensor by summing over all pointsource
        weihted by their scalar moment
        """
        x = 0.0
        y = 0.0
        z = 0.0
        finite_M0 = self.M0
        finite_mij = np.zeros(6)
        finite_time_shift = 0.0  # time shift is now included in the sliprate

        if dt is None:
            dt = self[0].dt

        # estimate the number of samples needed from the pointsource with
        # longest time_shift
        if nsamp is None:
            ps_ts_max = max(self.pointsources, key=lambda x: x.time_shift)
            nsamp = int(ps_ts_max.time_shift / dt + len(ps_ts_max.sliprate))

        finite_sliprate = np.zeros(nsamp)
        nfft = next_pow_2(nsamp) * 2
        self.resample_sliprate(dt, nsamp)

        for ps in self.pointsources:
            x += ps.x(planet_radius) * ps.M0 / finite_M0
            y += ps.y(planet_radius) * ps.M0 / finite_M0
            z += ps.z(planet_radius) * ps.M0 / finite_M0

            # finite_time_shift += ps.time_shift * ps.M0 / finite_M0

            mij = rotations.rotate_symm_tensor_voigt_xyz_src_to_xyz_earth(
                ps.tensor_voigt, np.deg2rad(ps.longitude),
                np.deg2rad(ps.colatitude))
            finite_mij += mij

            # sum sliprates with time shift applied
            sliprate_f = np.fft.rfft(ps.sliprate, n=nfft)
            sliprate_f *= np.exp(- 1j * rfftfreq(nfft) *
                                 2. * np.pi * ps.time_shift / dt)
            finite_sliprate += np.fft.irfft(sliprate_f)[:nsamp] \
                * ps.M0 / finite_M0

        longitude = np.rad2deg(np.arctan2(y, x))
        colatitude = np.rad2deg(
            np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2)))
        latitude = 90.0 - colatitude

        depth_in_m = planet_radius - (x ** 2 + y ** 2 + z ** 2) ** 0.5

        finite_mij = rotations.rotate_symm_tensor_voigt_xyz_earth_to_xyz_src(
            finite_mij, np.deg2rad(longitude), np.deg2rad(colatitude))

        self.CMT = Source(latitude, longitude, depth_in_m, m_rr=finite_mij[2],
                          m_tt=finite_mij[0], m_pp=finite_mij[1],
                          m_rt=finite_mij[4], m_rp=finite_mij[3],
                          m_tp=finite_mij[5], time_shift=finite_time_shift,
                          sliprate=finite_sliprate, dt=dt)

    @property
    def M0(self):
        """
        Scalar Moment M0 in Nm
        """
        return sum(ps.M0 for ps in self.pointsources)

    @property
    def moment_magnitude(self):
        """
        Moment magnitude M_w
        """
        return moment2magnitude(self.M0)

    @property
    def min_depth_in_m(self):
        return min(self.pointsources, key=lambda x: x.depth_in_m).depth_in_m

    @property
    def max_depth_in_m(self):
        return max(self.pointsources, key=lambda x: x.depth_in_m).depth_in_m

    @property
    def min_longitude(self):
        return min(self.pointsources, key=lambda x: x.longitude).longitude

    @property
    def max_longitude(self):
        return max(self.pointsources, key=lambda x: x.longitude).longitude

    @property
    def min_latitude(self):
        return min(self.pointsources, key=lambda x: x.latitude).latitude

    @property
    def max_latitude(self):
        return max(self.pointsources, key=lambda x: x.latitude).latitude

    @property
    def rupture_duration(self):
        ts_min = min(self.pointsources, key=lambda x: x.time_shift).time_shift
        ts_max = max(self.pointsources, key=lambda x: x.time_shift).time_shift
        return ts_max - ts_min

    @property
    def time_shift(self):
        return min(self.pointsources, key=lambda x: x.time_shift).time_shift

    @property
    def epicenter_latitude(self):
        return self.hypocenter_latitude

    @property
    def epicenter_longitude(self):
        return self.hypocenter_longitude

    @property
    def npointsources(self):
        return len(self.pointsources)

    def __str__(self):
        if (self.hypocenter_latitude is None and
                self.hypocenter_longitude) is None:
            self.find_hypocenter()

        return_str = 'Instaseis Finite Source:\n'
        return_str += '\tMoment Magnitude     : %4.2f\n' \
                      % (self.moment_magnitude)
        return_str += '\tscalar Moment        : %10.2e Nm\n' \
                      % (self.M0)
        return_str += '\t#point sources       : %d\n' \
                      % (self.npointsources)
        return_str += '\trupture duration     : %6.1f s\n' \
                      % (self.rupture_duration)
        return_str += '\ttime shift           : %6.1f s\n' \
                      % (self.time_shift)

        return_str += '\tmin depth            : %6.1f m\n' \
                      % (self.min_depth_in_m)
        return_str += '\tmax depth            : %6.1f m\n' \
                      % (self.max_depth_in_m)
        return_str += '\thypocenter depth     : %6.1f m\n' \
                      % (self.max_depth_in_m)

        return_str += '\tmin latitude         : %6.1f deg\n' \
                      % (self.min_latitude)
        return_str += '\tmax latitude         : %6.1f deg\n' \
                      % (self.max_latitude)
        return_str += '\thypocenter latitude  : %6.1f deg\n' \
                      % (self.hypocenter_latitude)

        return_str += '\tmin longitude        : %6.1f deg\n' \
                      % (self.min_longitude)
        return_str += '\tmax longitude        : %6.1f deg\n' \
                      % (self.max_longitude)
        return_str += '\thypocenter longitude : %6.1f deg\n' \
                      % (self.hypocenter_longitude)

        return return_str

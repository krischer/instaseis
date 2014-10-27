#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source and Receiver classes used for the AxiSEM DB Python interface.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import collections
import functools
import numpy as np
import obspy
import obspy.xseed
import os
from scipy import interp

from . import ReceiverParseError, SourceParseError

DEFAULT_MU = 32e9


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


class SourceOrReceiver(object):
    def __init__(self, latitude, longitude, depth_in_m):
        self.latitude = latitude
        self.longitude = longitude
        self.depth_in_m = depth_in_m

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
    A class to handle a seimic moment tensor source including a source time
    function.
    """
    def __init__(self, latitude, longitude, depth_in_m=None, m_rr=0.0,
                 m_tt=0.0, m_pp=0.0, m_rt=0.0, m_rp=0.0, m_tp=0.0,
                 time_shift=None, sliprate=None, dt=None):
        """
        :param latitude: latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param m_rr: moment tensor components in r, theta, phi in Nm
        :param m_tt: moment tensor components in r, theta, phi in Nm
        :param m_pp: moment tensor components in r, theta, phi in Nm
        :param m_rt: moment tensor components in r, theta, phi in Nm
        :param m_rp: moment tensor components in r, theta, phi in Nm
        :param m_tp: moment tensor components in r, theta, phi in Nm
        :param time_shift: correction of the origin time in seconds. only
            useful in the context of finite sources
        :param sliprate: normalized source time function (sliprate)
        :param dt: sampling of the source time function
        """
        super(Source, self).__init__(latitude, longitude, depth_in_m)
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp
        self.time_shift = time_shift
        self.sliprate = np.array(sliprate) if sliprate is not None else None
        self.dt = dt

    @staticmethod
    def parse(filename_or_obj):
        """
        Attempts to parse anything to a Source object. Can currently read
        anything ObsPy can read, ObsPy event related objects,
        and CMTSOLUTION files.

        For anything ObsPy related, it must contain a full moment tensor,
        otherwise it will raise an error.

        :param filename_or_obj: The object or filename to parse.
        """
        if isinstance(filename_or_obj, basestring):
            # Anything ObsPy can read.
            try:
                src = obspy.readEvents(filename_or_obj)
            except:
                pass
            else:
                return Source.parse(src)
            # CMT solution file.
            try:
                return Source.from_CMTSOLUTION_file(filename_or_obj)
            except:
                pass
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
            fm = ev.preferred_focal_mechanism() or ev.focal_mechansisms[0]
            if not fm.moment_tensor:
                raise SourceParseError("Event must contain a moment tensor.")
            t = fm.moment_tensor.tensor
            return Source(
                latitude=org.latitude,
                longitude=org.longitude,
                depth_in_m=org.depth,
                m_rr=t.m_rr,
                m_tt=t.m_tt,
                m_pp=t.m_pp,
                m_rt=t.m_rt,
                m_rp=t.m_rp,
                m_tp=t.m_tp)
        else:
            raise NotImplementedError

    @classmethod
    def from_CMTSOLUTION_file(self, filename):
        """
        Initialize a source object from a CMTSOLUTION file.

        :param filename: path to the CMTSOLUTION file
        """
        with open(filename, "rt") as f:
            f.readline()
            f.readline()
            time_shift = float(f.readline().strip().split()[-1])
            f.readline()
            latitude = float(f.readline().strip().split()[-1])
            longitude = float(f.readline().strip().split()[-1])
            depth_in_m = float(f.readline().strip().split()[-1]) * 1e3

            m_rr = float(f.readline().strip().split()[-1]) / 1e7
            m_tt = float(f.readline().strip().split()[-1]) / 1e7
            m_pp = float(f.readline().strip().split()[-1]) / 1e7
            m_rt = float(f.readline().strip().split()[-1]) / 1e7
            m_rp = float(f.readline().strip().split()[-1]) / 1e7
            m_tp = float(f.readline().strip().split()[-1]) / 1e7

        return self(latitude, longitude, depth_in_m, m_rr, m_tt, m_pp, m_rt,
                    m_rp, m_tp, time_shift)

    @classmethod
    def from_strike_dip_rake(self, latitude, longitude, depth_in_m, strike,
                             dip, rake, M0, time_shift=None, sliprate=None,
                             dt=None):
        """
        Initialize a source object from a shear source parameterized by strike,
        dip and rake.

        :param latitude: latitude of the source in degree
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

    def __str__(self):
        return_str = 'AxiSEM Database Source:\n'
        return_str += 'longitude : %6.1f deg\n' % (self.longitude)
        return_str += 'latitude  : %6.1f deg\n' % (self.latitude)
        return_str += 'Mrr       : %10.2e Nm\n' % (self.m_rr)
        return_str += 'Mtt       : %10.2e Nm\n' % (self.m_tt)
        return_str += 'Mpp       : %10.2e Nm\n' % (self.m_pp)
        return_str += 'Mrt       : %10.2e Nm\n' % (self.m_rt)
        return_str += 'Mrp       : %10.2e Nm\n' % (self.m_rp)
        return_str += 'Mtp       : %10.2e Nm\n' % (self.m_tp)

        return return_str


class ForceSource(SourceOrReceiver):
    """
    A class to handle a seimic force source.
    """
    def __init__(self, latitude, longitude, depth_in_m=None, f_r=0., f_t=0.,
                 f_p=0.):
        """
        :param latitude: latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param f_r: force components in r, theta, phi in N
        :param f_t: force components in r, theta, phi in N
        :param f_p: force components in r, theta, phi in N
        """
        super(ForceSource, self).__init__(latitude, longitude, depth_in_m)
        self.f_r = f_r
        self.f_t = f_t
        self.f_p = f_p

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
        return np.array([self.f_t, self.f_p, self.f_r])

    def __str__(self):
        return_str = 'AxiSEM Database Force Source:\n'
        return_str += 'longitude : %6.1f deg\n' % (self.longitude)
        return_str += 'latitude  : %6.1f deg\n' % (self.latitude)
        return_str += 'Fr        : %10.2e N\n' % (self.f_r)
        return_str += 'Ft        : %10.2e N\n' % (self.f_t)
        return_str += 'Fp        : %10.2e N\n' % (self.f_p)

        return return_str


class Receiver(SourceOrReceiver):
    """
    Class dealing with seismic receivers.
    """
    def __init__(self, latitude, longitude, network=None, station=None,
                 depth_in_m=None):
        """
        :param latitude: latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param network: network id
        :param station: station id
        """
        super(Receiver, self).__init__(latitude, longitude,
                                       depth_in_m=depth_in_m)
        self.network = network or ""
        self.station = station or ""

    def __str__(self):
        return_str = 'AxiSEM Database Receiver:\n'
        return_str += 'longitude : %6.1f deg\n' % (self.longitude)
        return_str += 'latitude  : %6.1f deg\n' % (self.latitude)
        return_str += 'name      : %s\n' % (self.station)
        return_str += 'network   : %s\n' % (self.network)

        return return_str

    @staticmethod
    @_purge_duplicates
    def parse(filename_or_obj, network_code=None):
        """
        Attempts to parse anything to a list of Receiver objects. Always
        returns a list, even if it only contains a single element. It is
        meant as a single entry point for receiver information from any source.

        Supports StationXML, the custom STATIONS fileformat, SAC files,
        SEED files, and a number of ObsPy objects. This method can
        furthermore work with anything ObsPy can deal with (filename, URL,
        memory files, ...).

        :param filename_or_obj: Filename/URL/Python object
        :param network_code: Network code needed to parse ObsPy station
            objects. Likely only needed for the recursive part of this method.
        :return: List of :class:`~instaseis.source.Receiver` objects.
        """
        receivers = []

        # STATIONS file.
        if isinstance(filename_or_obj, basestring) and \
                os.path.exists(filename_or_obj):
            try:
                return Receiver._parse_stations_file(filename_or_obj)
            except:
                pass
        # ObsPy inventory.
        elif isinstance(filename_or_obj, obspy.station.Inventory):
            for network in filename_or_obj:
                receivers.extend(Receiver.parse(network))
            return receivers
        # ObsPy network.
        elif isinstance(filename_or_obj, obspy.station.Network):
            for station in filename_or_obj:
                receivers.extend(Receiver.parse(
                    station, network_code=filename_or_obj.code))
            return receivers
        # ObsPy station.
        elif isinstance(filename_or_obj, obspy.station.Station):
            if network_code is None:
                raise ReceiverParseError("network_code must be given.")
            # If there are no channels, use the station coordinates.
            if not filename_or_obj.channels:
                return [Receiver(
                    latitude=filename_or_obj.latitude,
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
                return [Receiver(latitude=coords[0], longitude=coords[1],
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
            coords = (filename_or_obj.stats.sac.stla,
                      filename_or_obj.stats.sac.stlo)
            if -12345.0 in coords:
                raise ReceiverParseError(
                    "SAC file does not contain coordinates for channel '%s'" %
                    filename_or_obj.id)
            return [Receiver(latitude=coords[0], longitude=coords[1],
                             network=filename_or_obj.stats.network,
                             station=filename_or_obj.stats.station)]
        elif isinstance(filename_or_obj, obspy.xseed.Parser):
            inv = filename_or_obj.getInventory()
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
                receivers.append(Receiver(latitude=value[0][0],
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
        # Many StationXML files do not conform to the standard, thus the
        # ObsPy format detection fails. Catch those here.
        try:
            return Receiver.parse(obspy.read_inventory(filename_or_obj,
                                                       format="stationxml"))
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
            return Receiver.parse(obspy.xseed.Parser(filename_or_obj))
        except ReceiverParseError as e:
            raise e
        except:
            pass

        raise ValueError("'%s' could not be parsed." % repr(filename_or_obj))

    @staticmethod
    def _parse_stations_file(filename):
        """
        Parses a custom STATIONS file format to a list of Receiver objects.

        :param filename: Filename
        :return: List of :class:`~instaseis.source.Receiver` objects.
        """
        with open(filename, 'rt') as f:
            receivers = []

            for line in f:
                station, network, lat, lon, _, _ = line.split()
                lat = float(lat)
                lon = float(lon)
                receivers.append(Receiver(lat, lon, network, station))

        return receivers


class FiniteSource(object):
    """
    A class to handle finite sources represented my a number of point sources.
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

    def __iter__(self):
        return self

    def next(self):
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

        :param filename: path to the .srf file
        :param normalize: normalize the sliprate to 1
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
                rake, slip1, nt1, slip2, nt2, slip3, nt3 = \
                    map(float, f.readline().split())

                dep *= 1e3  # km    > m
                area *= 1e-4  # cm^2 > m^2
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

    def find_hypocenter(self):
        """
        Finds the hypo- and epicenter based on the point source that has the
        smallest timeshift
        """
        ps_hypo = min(self.pointsources, key=lambda x: x.time_shift)
        self.hypocenter_longitude = ps_hypo.longitude
        self.hypocenter_latitude = ps_hypo.latitude
        self.hypocenter_depth_in_m = ps_hypo.depth_in_m

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
    def epicenter_latitude(self):
        return self.hypocenter_latitude

    @property
    def epicenter_longitude(self):
        return self.hypocenter_longitude

    @property
    def npointsources(self):
        return len(self.pointsources)

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

import numpy as np
import obspy
from scipy import interp

EARTH_RADIUS = 6371.0 * 1000.0


class ReceiverParseError(Exception):
    pass


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
        latitude -- latitude of the source in degree
        longitude -- longitude of the source in degree
        depth_in_m -- source depth in m
        m_rr, m_tt, m_pp, m_rt, m_rp, m_tp -- moment tensor components in r,
        theta, phi coordinates in Nm
        time_shift -- correction of the origin time in seconds. only useful in
        the context of finite sources
        sliprate -- normalized source time function (sliprate)
        dt -- sampling of the source time function
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
        latitude -- latitude of the source in degree
        longitude -- longitude of the source in degree
        depth_in_m -- source depth in m
        strike -- strike of the fault in degree
        dip -- dip of the fault in degree
        rake -- rake of the fault in degree
        M0 -- scalar moment
        time_shift -- correction of the origin time in seconds. only useful in
        the context of finite sources
        sliprate -- normalized source time function (sliprate)
        dt -- sampling of the source time function
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
        sliprate -- (normalized) sliprate
        dt -- sampling of the sliprate
        normalize -- if sliprate is not normalized, set this to true to
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
        dt -- desired sampling
        nsamp -- desired number of samples
        """
        t_new = np.linspace(0, nsamp * dt, nsamp, endpoint=False)
        t_old = np.linspace(0, self.dt * len(self.sliprate),
                            len(self.sliprate), endpoint=False)

        self.sliprate = interp(t_new, t_old, self.sliprate)
        self.dt = dt


class Receiver(SourceOrReceiver):
    """
    Class dealing with seismic receivers.
    """
    def __init__(self, latitude, longitude, network=None, station=None):
        """
        latitude -- latitude of the source in degree
        longitude -- longitude of the source in degree
        network -- network id
        station -- station id
        """
        super(Receiver, self).__init__(latitude, longitude, depth_in_m=0.0)
        self.network = network or ""
        self.station = station or ""

    @staticmethod
    def parse(filename_or_obj, network_code=None):
        """
        Attempts to parse anything to a list of Receiver objects. Always
        returns a list, even if it only contains a single element. It is
        meant as a single entry point for receiver information from any source.

        Supports StationXML, the custom STATIONS fileformat, SAC files,
        and a number of ObsPy objects. This method can furthermore work with
        anything ObsPy can deal with (filename, URL, memory files, ...).

        :param filename_or_obj: Filename/URL/Python object
        :param network_code: Network code needed to parse ObsPy station
            objects. Likely only needed for the recursive part of this method.
        :return: List of :class:`~axisem_db.source.Receiver` objects.
        """
        receivers = []

        # STATIONS file.
        if isinstance(filename_or_obj, basestring):
            try:
                return Receiver.parse_stations_file(filename_or_obj)
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
        try:
            return Receiver.parse(obspy.read(filename_or_obj))
        except ReceiverParseError as e:
            raise e
        except:
            pass

        raise ValueError("'%s' could not be parsed." % repr(filename_or_obj))

    @staticmethod
    def parse_stations_file(filename):
        """
        Parses a custom STATIONS file format to a list of Receiver objects.

        :param filename: Filename
        :return: List of :class:`~axisem_db.source.Receiver` objects.
        """
        with open(filename, 'rt') as f:
            receivers = []

            for line in f:
                station, network, lat, lon, _, _ = line.split()
                lat = float(lat)
                lon = float(lon)
                receivers.append(Receiver(lat, lon, network, station))

        return receivers

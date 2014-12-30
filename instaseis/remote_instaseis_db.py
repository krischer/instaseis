#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instaseis database class for remote access over HTTP.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library
with standard_library.hooks():
    from urllib.parse import urlunparse, urlencode, urlparse

import obspy
import requests

from . import InstaseisError, Source, ForceSource
from .base_instaseis_db import BaseInstaseisDB


class RemoteInstaseisDB(BaseInstaseisDB):
    """
    Remote instaseis database interface.
    """
    def __init__(self, url):
        """
        :param url: URL to the remote instaseis server.
        :type db_path: str
        """
        self.url = url
        self._scheme, self._netloc = urlparse(url)[:2]

        # Parse the root message of the server.
        try:
            root = self._download_url(self._get_url(path=""), unpack_json=True)
        except Exception as e:
            raise InstaseisError("Failed to connect to remote Instaseis "
                                 "server due to: %s" % (str(e)))

        # XXX: Add Instaseis version checks! Make sure server and client are
        # on the same version!
        if "type" not in root or root["type"] != "Instaseis Remote Server":
            raise InstaseisError("Instaseis server responded with invalid "
                                 "response: %s" % (str(root)))
        self.get_info()

    def get_seismograms(self, source, receiver, components=("Z", "N", "E"),
                        kind='displacement', remove_source_shift=True,
                        reconvolve_stf=False, return_obspy_stream=True,
                        dt=None, a_lanczos=5):
        """
        Extract seismograms for a moment tensor point source from the AxiSEM
        database.

        :param source: instaseis.Source or instaseis.ForceSource object
        :type source: :class:`instaseis.source.Source` or
            :class:`instaseis.source.ForceSource`
        :param receiver: instaseis.Receiver object
        :type receiver: :class:`instaseis.source.Receiver`
        :param components: a tuple containing any combination of the
            strings ``"Z"``, ``"N"``, ``"E"``, ``"R"``, and ``"T"``
        :param kind: 'displacement', 'velocity' or 'acceleration'
        :param remove_source_shift: move the starttime to the peak of the
            sliprate from the source time function used to generate the
            database
        :param reconvolve_stf: deconvolve the source time function used in
            the AxiSEM run and convolve with the stf attached to the source.
            For this to be stable, the new stf needs to bandlimited.
        :param return_obspy_stream: return format is either an obspy.Stream
            object or a plain array containing the data
        :param dt: desired sampling of the seismograms. resampling is done
            using a lanczos kernel
        :param a_lanczos: width of the kernel used in resampling
        """
        source, receiver = self._get_seismograms_sanity_checks(
            source=source, receiver=receiver, components=components, kind=kind)

        # Collect parameters.
        params = {}

        # Start with the receiver.
        params["receiver_latitude"] = receiver.latitude
        params["receiver_longitude"] = receiver.longitude
        if receiver.depth_in_m is not None:
            params["receiver_depth_in_m"] = receiver.depth_in_m
        if receiver.network:
            params["network_code"] = receiver.network
        if receiver.station:
            params["station_code"] = receiver.station

        # Do the source.
        params["source_latitude"] = source.latitude
        params["source_longitude"] = source.longitude
        if source.depth_in_m is not None:
            params["source_depth_in_m"] = source.depth_in_m
        if isinstance(source, ForceSource):
            params["f_r"] = source.f_r
            params["f_t"] = source.f_t
            params["f_p"] = source.f_p
        elif isinstance(source, Source):
            params["m_rr"] = source.m_rr
            params["m_tt"] = source.m_tt
            params["m_pp"] = source.m_pp
            params["m_rt"] = source.m_rt
            params["m_rp"] = source.m_rp
            params["m_tp"] = source.m_tp
        else:
            raise NotImplementedError

        url = self._get_url(path="seismograms", **params)
        # Just utilize ObsPy to download and unpack the data!
        st = obspy.read(url)
        return st

    def _get_url(self, path, **kwargs):
        return urlunparse((self._scheme, self._netloc, path, None,
                           urlencode(kwargs), None))

    def _download_url(self, url, unpack_json=False):
        """
        Helper function downloading data from a URL.
        """
        r = requests.get(url)
        if r.status_code != 200:
            raise InstaseisError("Status code %i when downloading '%s'" % (
                r.status_code, url))
        if unpack_json is True:
            return r.json()
        else:
            return r.text

    def get_info(self):
        """
        Returns a dictionary with information about the currently loaded
        database.
        """
        info = self._download_url(self._get_url(path="info"), unpack_json=True)
        info["directory"] = self.url
        return info

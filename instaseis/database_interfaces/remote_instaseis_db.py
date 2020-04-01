#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instaseis database class for remote access over HTTP.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import io
import numpy as np
import obspy
from urllib.parse import urlencode, urlparse
import requests
import warnings

from .base_instaseis_db import BaseInstaseisDB, DEFAULT_MU
from .. import (
    InstaseisError,
    InstaseisWarning,
    Source,
    ForceSource,
    __version__,
)


class RemoteInstaseisDB(BaseInstaseisDB):
    """
    Remote Instaseis database interface.
    """

    def __init__(self, url, *args, **kwargs):
        """
        :param url: URL to the remote Instaseis server.
        :type db_path: str
        """
        self.url = url
        self._scheme, self._netloc, self._path = urlparse(url)[:3]
        self._path = self._path.strip("/")

        # Parse the root message of the server.
        try:
            root = self._download_url(self._get_url(path=""))
        except Exception as e:
            raise InstaseisError(
                "Failed to connect to remote Instaseis "
                "server due to: %s" % (str(e))
            )

        # XXX: Add Instaseis version checks! Make sure server and client are
        # on the same version!
        if "type" not in root or root["type"] != "Instaseis Remote Server":
            raise InstaseisError(
                "Instaseis server responded with invalid "
                "response: %s" % (str(root))
            )

        if root["version"] != __version__:
            msg = (
                "Instaseis versions on server (%s) and on your local "
                "client (%s) differ and thus things might not work as "
                "expected." % (root["version"], __version__)
            )
            warnings.warn(msg, InstaseisWarning)
        self._get_info()

    def _get_seismograms(self, source, receiver, components=("Z", "N", "E")):
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
        """
        # Collect parameters.
        params = {"components": "".join(components).upper()}

        # Start with the receiver.
        params["receiverlatitude"] = receiver.latitude
        params["receiverlongitude"] = receiver.longitude
        if receiver.depth_in_m is not None:
            params["receiverdepthinmeters"] = receiver.depth_in_m
        if receiver.network:
            params["networkcode"] = receiver.network
        if receiver.station:
            params["stationcode"] = receiver.station

        # Do the source.
        params["sourcelatitude"] = source.latitude
        params["sourcelongitude"] = source.longitude
        if source.depth_in_m is not None:
            params["sourcedepthinmeters"] = source.depth_in_m
        if isinstance(source, ForceSource):
            params["fr"] = source.f_r
            params["ft"] = source.f_t
            params["fp"] = source.f_p
        elif isinstance(source, Source):
            params["mrr"] = source.m_rr
            params["mtt"] = source.m_tt
            params["mpp"] = source.m_pp
            params["mrt"] = source.m_rt
            params["mrp"] = source.m_rp
            params["mtp"] = source.m_tp
        else:
            raise NotImplementedError

        url = self._get_url(path="seismograms_raw", **params)

        r = requests.get(url)
        if "Instaseis-Mu" not in r.headers:  # pragma: no cover
            warnings.warn(
                "Mu is not passed via the HTTP headers. Maybe some "
                "proxy removed it? Mu is now always the default mu.",
                InstaseisWarning,
            )
            mu = DEFAULT_MU
        else:
            mu = float(r.headers["Instaseis-Mu"])

        with io.BytesIO(r.content) as fh:
            fh.seek(0, 0)
            st = obspy.read(fh)

        # Convert back to dictionary of numpy arrays...this is a bit
        # redundant but plays nice with the rest of Instaseis and still
        # enables a REST API that serves MiniSEED files.
        data = {"mu": mu}

        for tr in st:
            data[tr.stats.channel[-1].upper()] = tr.data

        return data

    def _get_url(self, path, **kwargs):
        # Not tested in the test-suite as it would be awkward to do with the
        # current setup. But manually vetted and should be good.
        if self._path:  # pragma: no cover
            path = "/" + self._path + "/" + path

        url = "%s://%s" % (self._scheme, self._netloc)
        if path:
            url += "/%s" % path
        if kwargs:
            url += "?%s" % urlencode(kwargs)
        return url

    def _download_url(self, url):
        """
        Helper function downloading data from a URL.
        """
        r = requests.get(url)
        # Not tested in test suite as it would be awkward to do. Manually
        # tested and should be good.
        if r.status_code != 200:  # pragma: no cover
            raise InstaseisError(
                "Status code %i when downloading '%s'" % (r.status_code, url)
            )
        return r.json()

    def _get_info(self):
        """
        Returns a dictionary with information about the currently loaded
        database.
        """
        info = self._download_url(self._get_url(path="info"))
        info["directory"] = self.url
        # Convert types lost in the translation to JSON.
        info["datetime"] = obspy.UTCDateTime(info["datetime"])
        info["slip"] = np.array(info["slip"], dtype=np.float64)
        info["sliprate"] = np.array(info["sliprate"], dtype=np.float64)

        return info

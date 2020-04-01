#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instaseis database class for remote access using the syngine service of IRIS.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import io
import numpy as np
import obspy
import platform
import requests
from urllib.parse import urlencode
import warnings

from instaseis import (
    InstaseisError,
    InstaseisWarning,
    Source,
    ForceSource,
    __version__,
)
from instaseis.database_interfaces.base_instaseis_db import (
    BaseInstaseisDB,
    DEFAULT_MU,
    STF_MAP,
    INV_KIND_MAP,
)

from instaseis.helpers import geocentric_to_elliptic_latitude


USER_AGENT = "Instaseis %s (%s, Python %s)" % (
    __version__,
    platform.platform(),
    platform.python_version(),
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip,deflate"}


class SyngineInstaseisDB(BaseInstaseisDB):
    """
    Remote Instaseis interface connecting with IRIS' syngine service.
    """

    def __init__(
        self,
        model,
        base_url="http://service.iris.edu/irisws/syngine/1",
        debug=False,
        *args,
        **kwargs,
    ):
        """
        :param model: The model to use.
        :type model: str
        :param base_url: URL to the root of the syngine service.
        :type base_url: str
        :param debug: Debug messages on/off.
        :type debug: bool
        """
        self.model = model
        self.debug = debug
        self.base_url = base_url.rstrip("/")

        # Download once to make sure it works and the model exists.
        self.info

        # Get the version of the service.
        self.syngine_service_version = self._download_url(
            self._get_url(path="version")
        )

    def _get_seismograms(self, source, receiver, components=("Z", "N", "E")):
        """
        Extract seismograms.

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

        params["model"] = self.model
        params["format"] = "miniseed"

        # Always request in the original units.
        params["units"] = INV_KIND_MAP[STF_MAP[self.info.stf]]

        # No resampling!
        params["dt"] = self.info.dt

        # Carefully set the starttime to get everything. This is a bit
        # awkward and we essentially have to undo what the /seismograms
        # route is doing...
        params["origintime"] = str(obspy.UTCDateTime(0))
        params["starttime"] = str(
            obspy.UTCDateTime(-(self.info.dt * (self.info.src_shift_samples)))
        )

        # Start with the receiver.
        # syngine uses WGS84 coordinates - we assume geocentric ones. Thus
        # we have to convert.
        params["receiverlatitude"] = geocentric_to_elliptic_latitude(
            receiver.latitude
        )
        params["receiverlongitude"] = receiver.longitude
        if receiver.depth_in_m:  # pragma: no cover
            warnings.warn(
                "The syngine service only services reciprocal "
                "databases thus the receiver depth cannot be "
                "changed.",
                UserWarning,
            )
        # The syngine services requires a station code.
        params["stationcode"] = "SYN"

        # Do the source.
        params["sourcelatitude"] = geocentric_to_elliptic_latitude(
            source.latitude
        )
        params["sourcelongitude"] = source.longitude
        if source.depth_in_m is not None:
            params["sourcedepthinmeters"] = source.depth_in_m
        if isinstance(source, ForceSource):
            raise ValueError(
                "The Syngine Instaseis client does currently not "
                "support force sources. You can still download "
                "data from the Syngine service for force "
                "sources manually."
            )
        elif isinstance(source, Source):
            params["sourcemomenttensor"] = ",".join(
                map(
                    str,
                    [
                        source.m_rr,
                        source.m_tt,
                        source.m_pp,
                        source.m_rt,
                        source.m_rp,
                        source.m_tp,
                    ],
                )
            )
        else:
            raise NotImplementedError

        url = self._get_url(path="query", **params)

        if self.debug:  # pragma: no cover
            print("Downloading '%s' ..." % url)

        r = requests.get(url, headers=HEADERS)

        if self.debug:  # pragma: no cover
            print(
                "Downloaded '%s' with status code %i." % (url, r.status_code)
            )

        if r.status_code != 200:  # pragma: no cover
            # Right now one has to parse the body of the response to get the
            # message.
            reason = r.content.decode().strip().split("\n")[0]
            raise InstaseisError(
                "Status code %i when downloading '%s'. Reason: '%s'"
                % (r.status_code, url, reason)
            )

        if "instaseis-mu" not in r.headers:  # pragma: no cover
            warnings.warn(
                "Mu is not passed via the HTTP headers. Maybe some "
                "proxy removed it? Mu is now always the default mu.",
                InstaseisWarning,
            )
            mu = DEFAULT_MU
        else:  # pragma: no cover
            mu = float(r.headers["instaseis-mu"])

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
        path = path.strip("/")
        url = "%s/%s" % (self.base_url, path)

        if kwargs:
            url += "?%s" % urlencode(kwargs)
        return url

    def _download_url(self, url, unpack_json=False):
        """
        Helper function downloading data from a URL.
        """
        if self.debug:  # pragma: no cover
            print("Downloading '%s' ..." % url)
        r = requests.get(url, headers=HEADERS)
        if self.debug:  # pragma: no cover
            print(
                "Downloaded '%s' with status code %i." % (url, r.status_code)
            )
        if r.status_code == 400:  # pragma: no cover
            raise InstaseisError(
                "Model '%s' not available on the syngine "
                "service?" % self.model
            )
        elif r.status_code != 200:  # pragma: no cover
            raise InstaseisError(
                "Status code %i when downloading '%s'" % (r.status_code, url)
            )
        if unpack_json is True:
            return r.json()
        else:
            return r.text

    def _get_info(self):
        """
        Returns a dictionary with information about the currently loaded
        database.
        """
        info = self._download_url(
            self._get_url(path="info", model=self.model), unpack_json=True
        )
        info["directory"] = self.base_url
        # Convert types lost in the translation to JSON.
        info["datetime"] = obspy.UTCDateTime(info["datetime"])
        info["slip"] = np.array(info["slip"], dtype=np.float64)
        info["sliprate"] = np.array(info["sliprate"], dtype=np.float64)

        return info

    def __str__(self):
        base = BaseInstaseisDB.__str__(self).splitlines()
        base.insert(1, "Syngine model name:      '%s'" % self.model)
        base.insert(
            2, "Syngine service version:  %s" % self.syngine_service_version
        )
        return "\n".join(base)

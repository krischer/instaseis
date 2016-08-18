#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import io
import math
import re
import functools
import threading

import numpy as np
import obspy
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.io.sac.util import utcdatetime_to_sac_nztimes
import tornado.web

from .. import ForceSource, FiniteSource
from ..helpers import geocentric_to_elliptic_latitude
from .. import __version__

# Valid phase offset pattern including capture groups.
PHASE_OFFSET_PATTERN = re.compile(r"(^[A-Za-z0-9^]+)([\+-])([\deE\.\-\+]+$)")


def run_async(func):
    """
    Decorator executing a function in a thread.

    Adapted from http://stackoverflow.com/a/15952516/1657047
    """
    @functools.wraps(func)
    def async_func(*args, **kwargs):
        func_hl = threading.Thread(target=func, args=args, kwargs=kwargs)
        func_hl.start()
        return func_hl
    return async_func


class IOQueue(object):
    """
    Object passed to the zipfile constructor which acts as a file-like object.

    Iterating over the object yields the data pieces written to it since it
    has last been iterated over DELETING those pieces at the end of each
    loop. This enables the server to send unbounded zipfiles without running
    into memory issues.
    """
    def __init__(self):
        self.count = 0
        self.data = []

    def flush(self):
        pass

    def tell(self):
        return self.count

    def write(self, data):
        self.data.append(data)
        self.count += len(data)

    def __iter__(self):
        for _i in self.data:
            yield _i
        self.data = []
        raise StopIteration


def _validtimesetting(value):
    try:
        return obspy.UTCDateTime(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    m = PHASE_OFFSET_PATTERN.match(value)
    if m is None:
        raise ValueError

    operator = m.group(2)
    if operator == "+":
        offset = float(m.group(3))
    else:
        offset = -float(m.group(3))

    return {
        "phase": m.group(1),
        "offset": offset
    }


def _format_utc_datetime(dt):
    """
    Python 2's datetime class cannot format dates before 1900. Thus we do it
    like this which yields the same result.
    """
    return dt.datetime.isoformat() + "Z"


def _validate_and_write_waveforms(st, callback, starttime, endtime, scale,
                                  source, receiver, db, label, format):
    if not label:
        label = ""
    else:
        label += "_"

    for tr in st:
        # Half the filesize but definitely sufficiently accurate.
        tr.data = np.require(tr.data, dtype=np.float32)

    if scale != 1.0:
        for tr in st:
            tr.data *= scale

    # Sanity checks. Raise internal server errors in case something fails.
    # This should not happen and should have been caught before.
    if endtime > st[0].stats.endtime:
        msg = ("Endtime larger than the extracted endtime: endtime=%s, "
               "largest db endtime=%s" % (
                _format_utc_datetime(endtime),
                _format_utc_datetime(st[0].stats.endtime)))
        callback((tornado.web.HTTPError(500, log_message=msg, reason=msg),
                  None))
        return
    if starttime < st[0].stats.starttime - 3600.0:
        msg = ("Starttime more than one hour before the starttime of the "
               "seismograms.")
        callback((tornado.web.HTTPError(500, log_message=msg, reason=msg),
                  None))
        return

    if isinstance(source, FiniteSource):
        mu = None
    else:
        mu = st[0].stats.instaseis.mu

    # Trim, potentially pad with zeroes.
    st.trim(starttime, endtime, pad=True, fill_value=0.0, nearest_sample=False)

    # Checked in another function and just a sanity check.
    assert format in ("miniseed", "saczip")

    if format == "miniseed":
        with io.BytesIO() as fh:
            st.write(fh, format="mseed")
            fh.seek(0, 0)
            binary_data = fh.read()
        callback((binary_data, mu))
    # Write a number of SAC files into an archive.
    elif format == "saczip":
        byte_strings = []
        for tr in st:
            # Write SAC headers.
            tr.stats.sac = obspy.core.AttribDict()
            # Write WGS84 coordinates to the SAC files.
            tr.stats.sac.stla = geocentric_to_elliptic_latitude(
                receiver.latitude)
            tr.stats.sac.stlo = receiver.longitude
            tr.stats.sac.stdp = receiver.depth_in_m
            tr.stats.sac.stel = 0.0
            if isinstance(source, FiniteSource):
                tr.stats.sac.evla = geocentric_to_elliptic_latitude(
                    source.hypocenter_latitude)
                tr.stats.sac.evlo = source.hypocenter_longitude
                tr.stats.sac.evdp = source.hypocenter_depth_in_m
                # Force source has no magnitude.
                if not isinstance(source, ForceSource):
                    tr.stats.sac.mag = source.moment_magnitude
                src_lat = source.hypocenter_latitude
                src_lng = source.hypocenter_longitude
            else:
                tr.stats.sac.evla = geocentric_to_elliptic_latitude(
                    source.latitude)
                tr.stats.sac.evlo = source.longitude
                tr.stats.sac.evdp = source.depth_in_m
                # Force source has no magnitude.
                if not isinstance(source, ForceSource):
                    tr.stats.sac.mag = source.moment_magnitude
                src_lat = source.latitude
                src_lng = source.longitude
            # Thats what SPECFEM uses for a moment magnitude....
            tr.stats.sac.imagtyp = 55
            # The event origin time relative to the reference which I'll
            # just assume to be the starttime here?
            tr.stats.sac.o = source.origin_time - starttime

            # Sac coordinates are elliptical thus it only makes sense to
            # have elliptical distances.
            dist_in_m, az, baz = gps2dist_azimuth(
                lat1=tr.stats.sac.evla,
                lon1=tr.stats.sac.evlo,
                lat2=tr.stats.sac.stla,
                lon2=tr.stats.sac.stlo)

            tr.stats.sac.dist = dist_in_m / 1000.0
            tr.stats.sac.az = az
            tr.stats.sac.baz = baz

            # XXX: Is this correct? Maybe better use some function in
            # geographiclib?
            tr.stats.sac.gcarc = locations2degrees(
                lat1=src_lat,
                long1=src_lng,
                lat2=receiver.latitude,
                long2=receiver.longitude)

            # Set two more headers. See #45.
            tr.stats.sac.lpspol = 1
            tr.stats.sac.lcalda = 0

            # Some provenance.
            tr.stats.sac.kuser0 = "InstSeis"
            tr.stats.sac.kuser1 = db.info.velocity_model[:8]
            tr.stats.sac.user0 = scale
            # Prefix version numbers to identify them at a glance.
            tr.stats.sac.kt7 = "A" + db.info.axisem_version[:7]
            tr.stats.sac.kt8 = "I" + __version__[:7]

            # Times have to be set by hand.
            t, _ = utcdatetime_to_sac_nztimes(tr.stats.starttime)
            for key, value in t.items():
                tr.stats.sac[key] = value

            with io.BytesIO() as temp:
                tr.write(temp, format="sac")
                temp.seek(0, 0)
                filename = "%s%s.sac" % (label, tr.id)
                byte_strings.append((filename, temp.read()))
        callback((byte_strings, mu))


def get_gaussian_source_time_function(source_width, dt):
    """
    Returns a gaussian source time function.

    :type source_width: float
    :param source_width: The desired source width in seconds. This is twice
        the half-duration as used in many waveform solvers.
    :type dt: float
    :param dt: The sample interval of the STF.

    Returns a tuple with two things:

    1. The offset from the first sample to the peak of the gaussian in
        seconds. This is guaranteed to be directly on a sample.
    2. The actual source time function as a numpy array.

    It is normalized to zero and first and last sample are also guaranteed
    to be zero.
    """
    # We calculate it for twice the source width, and set the
    # offset to the next sample.
    x = int(math.ceil(source_width / dt))
    offset = x * dt
    t = np.linspace(0, 2 * offset + dt, x * 2 + 2)

    # Sanity check.
    assert np.isclose(t[1] - t[0], dt)

    a = 1.0 / ((0.25 * source_width) ** 2)

    y = np.exp(-a * (t - offset) ** 2) / (np.sqrt(np.pi) * 0.25 * source_width)

    # Sanity checks and manually set the first and last sample to 0.
    y_m = y.max()
    assert y[0] <= 1E-5 * y_m and y[-1] <= 1E-5 * y_m
    y[0] = 0
    y[-1] = 0

    return offset, y

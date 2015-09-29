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
import re
import functools
import threading

import numpy as np
import obspy
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


def _validate_and_write_waveforms(st, callback, starttime, endtime, source,
                                  receiver, db, label, format):
    if not label:
        label = ""
    else:
        label += "_"

    for tr in st:
        # Half the filesize but definitely sufficiently accurate.
        tr.data = np.require(tr.data, dtype=np.float32)
    # Sanity checks. Raise internal server errors in case something fails.
    # This should not happen and should have been caught before.
    if endtime > st[0].stats.endtime:
        msg = ("Endtime larger than the extracted endtime: endtime=%s, "
               "largest db endtime=%s" % (endtime, st[0].stats.endtime))
        callback(tornado.web.HTTPError(500, log_message=msg, reason=msg))
        return
    if starttime < st[0].stats.starttime - 3600.0:
        msg = ("Starttime more than one hour before the starttime of the "
               "seismograms.")
        callback(tornado.web.HTTPError(500, log_message=msg, reason=msg))
        return

    # Trim, potentially pad with zeroes.
    st.trim(starttime, endtime, pad=True, fill_value=0.0, nearest_sample=False)

    # Checked in another function and just a sanity check.
    assert format in ("miniseed", "saczip")

    if format == "miniseed":
        with io.BytesIO() as fh:
            st.write(fh, format="mseed")
            fh.seek(0, 0)
            binary_data = fh.read()
        callback(binary_data)
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
            else:
                tr.stats.sac.evla = geocentric_to_elliptic_latitude(
                    source.latitude)
                tr.stats.sac.evlo = source.longitude
                tr.stats.sac.evdp = source.depth_in_m
                # Force source has no magnitude.
                if not isinstance(source, ForceSource):
                    tr.stats.sac.mag = source.moment_magnitude
            # Thats what SPECFEM uses for a moment magnitude....
            tr.stats.sac.imagtyp = 55
            # The event origin time relative to the reference which I'll
            # just assume to be the starttime here?
            tr.stats.sac.o = source.origin_time - starttime
            # Some provenance.
            tr.stats.sac.kuser0 = "InstSeis"
            tr.stats.sac.kuser1 = __version__[:8]
            tr.stats.sac.kuser2 = db.info.velocity_model[:8]

            with io.BytesIO() as temp:
                tr.write(temp, format="sac")
                temp.seek(0, 0)
                filename = "%s%s.sac" % (label, tr.id)
                byte_strings.append((filename, temp.read()))
        callback(byte_strings)

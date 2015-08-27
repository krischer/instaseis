#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Martin van Driel (vandriel@erdw.ethz.ch), 2015
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import io
import zipfile

import numpy as np
import obspy
import tornado.gen
import tornado.web

from ... import Source, Receiver
from ..util import run_async, _validtimesetting
from ..instaseis_request import InstaseisTimeSeriesHandler
from ...helpers import geocentric_to_elliptic_latitude


@run_async
def _get_greens(db, epicentral_distance_degree, source_depth_in_m, units, dt,
                kernelwidth, origintime, starttime, endtime, format, label,
                callback):
    """
    Extract a Green's function from the passed db and write it either to a
    MiniSEED or a SACZIP file.

    :param db: An open instaseis database.
    :param epicentral_distance_degree: epicentral distance in degree
    :param source_depth_in_m: source depth in m
    :param units: The desired units.
    :param dt: dt to resample to.
    :param kernelwidth: Width of the interpolation kernel.
    :param origintime: Origin time of the source.
    :param starttime: The desired start time of the seismogram.
    :param endtime: The desired end time of the seismogram.
    :param format: The output format. Either "miniseed" or "saczip".
    :param label: Prefix for the filename within the SAC zip file.
    :param callback: callback function of the coroutine.
    """
    if not label:
        label = ""
    else:
        label += "_"

    try:
        st = db.get_greens_function(
            epicentral_distance_in_degree=epicentral_distance_degree,
            source_depth_in_m=source_depth_in_m, origin_time=origintime,
            kind=units, return_obspy_stream=True, dt=dt,
            kernelwidth=kernelwidth, definition="seiscomp")
    except Exception:
        msg = ("Could not extract Green's function. Make sure, the parameters "
               "are valid, and the depth settings are correct.")
        callback(tornado.web.HTTPError(400, log_message=msg, reason=msg))
        return

    for tr in st:
        # Half the filesize but definitely sufficiently accurate.
        tr.data = np.require(tr.data, dtype=np.float32)

    # Sanity checks. Raise internal server errors in case something fails.
    # This should not happen and should have been caught before.
    if endtime > st[0].stats.endtime:
        msg = ("Endtime larger then the extracted endtime: endtime=%s, "
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
                90.0 - epicentral_distance_degree)
            tr.stats.sac.stlo = 0.0
            tr.stats.sac.stdp = 0.0
            tr.stats.sac.stel = 0.0
            tr.stats.sac.evla = 90.0
            tr.stats.sac.evlo = 90.0
            tr.stats.sac.evdp = source_depth_in_m
            # Thats what SPECFEM uses for a moment magnitude....
            tr.stats.sac.imagtyp = 55
            # The event origin time relative to the reference which I'll
            # just assume to be the starttime here?
            tr.stats.sac.o = origintime - starttime

            with io.BytesIO() as temp:
                tr.write(temp, format="sac")
                temp.seek(0, 0)
                filename = "%s%s.sac" % (label, tr.id)
                byte_strings.append((filename, temp.read()))
        callback(byte_strings)


class GreensFunctionHandler(InstaseisTimeSeriesHandler):
    # Define the arguments for the Greens endpoint.
    arguments = {
        "units": {"type": str, "default": "displacement"},
        "dt": {"type": float},
        "kernelwidth": {"type": int, "default": 12},
        "label": {"type": str},

        # Source parameters.
        "sourcedistanceindegrees": {"type": float, "required": True},
        "sourcedepthinmeters": {"type": float, "required": True},

        # Time parameters.
        "origintime": {"type": obspy.UTCDateTime},
        "starttime": {"type": _validtimesetting,
                      "format": "Datetime String/Float/Phase+-Offset"},
        "endtime": {"type": _validtimesetting,
                    "format": "Datetime String/Float/Phase+-Offset"},

        "format": {"type": str, "default": "saczip"}
    }

    default_label = "instaseis_greens_function"

    def __init__(self, *args, **kwargs):
        super(InstaseisTimeSeriesHandler, self).__init__(*args, **kwargs)

    def validate_parameters(self, args):
        """
        Function attempting to validate that the passed parameters are
        valid. Does not need to check the types as that has already been done.
        """
        info = self.application.db.info

        # greens functions only work with reciprocal databases
        if not info.is_reciprocal:
            msg = ("The database is not reciprocal, "
                   "so Green's functions can't be computed.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure thaat epicentral disance and source depth are in reasonable
        # ranges
        if args.sourcedistanceindegrees is not None and \
                not 0.0 <= args.sourcedistanceindegrees <= 180.0:
            msg = ("Epicentral distance should be in [0, 180].")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        min_depth = info.planet_radius - info.max_radius
        max_depth = info.planet_radius - info.min_radius
        if args.sourcedepthinmeters is not None and \
                not min_depth <= args.sourcedepthinmeters <= max_depth:
            msg = ("Source depth should be in [%.1f, %.1f]." % (
                   min_depth, max_depth))
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        # Parse the arguments. This will also perform a number of sanity
        # checks.
        args = self.parse_arguments()

        min_starttime, max_endtime = self.parse_time_settings(args)

        self.set_headers(args)

        # generating source and receiver as in the get_greens routine of the
        # base_instaseis class
        src_latitude, src_longitude = 90., 0.
        rec_latitude, rec_longitude = 90. - args.sourcedistanceindegrees, 0.
        source = Source(src_latitude, src_longitude, args.sourcedepthinmeters)
        receiver = Receiver(rec_latitude, rec_longitude)

        # Validate the source-receiver geometry.
        self.validate_geometry(source=source, receiver=receiver)

        # Get phase-relative times.
        time_values = self.get_phase_relative_times(
            args=args, source=source, receiver=receiver,
            min_starttime=min_starttime, max_endtime=max_endtime)

        if time_values is None:
            msg = ("No Green's function extracted for the given phase "
                   "relative offsets. This could either be due to the "
                   "chosen phase not existing for the specific "
                   "source-receiver geometry or arriving too late/with "
                   "too large offsets if the database is not long enough.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        starttime, endtime = time_values

        # Yield from the task. This enables a context switch and thus
        # async behaviour.
        response = yield tornado.gen.Task(
            _get_greens,
            db=self.application.db,
            epicentral_distance_degree=args.sourcedistanceindegrees,
            source_depth_in_m=args.sourcedepthinmeters, units=args.units,
            dt=args.dt, kernelwidth=args.kernelwidth,
            origintime=args.origintime, starttime=starttime,
            endtime=endtime, format=args.format, label=args.label)

        # If an exception is returned from the task, re-raise it here.
        if isinstance(response, Exception):
            raise response

        if args.format == "miniseed":
            self.write(response)
        else:
            assert args.format == "saczip"
            assert isinstance(response, list)

            with io.BytesIO() as buf:
                zip_file = zipfile.ZipFile(buf, mode="w")
                for filename, content in response:
                    zip_file.writestr(filename, content)
                zip_file.close()
                buf.seek(0, 0)
                self.write(buf.read())

        self.finish()

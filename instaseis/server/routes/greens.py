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

import obspy
import tornado.gen
import tornado.web

from ... import Source, Receiver, ForceSource
from ..util import run_async, _validtimesetting, _validate_and_write_waveforms
from ..instaseis_request import InstaseisTimeSeriesHandler


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
    try:
        st = db.get_greens_function(
            epicentral_distance_in_degree=epicentral_distance_degree,
            source_depth_in_m=source_depth_in_m, origin_time=origintime,
            kind=units, return_obspy_stream=True, dt=dt,
            kernelwidth=kernelwidth, definition="seiscomp")
    except Exception:
        msg = ("Could not extract Green's function. Make sure, the parameters "
               "are valid, and the depth settings are correct.")
        callback((tornado.web.HTTPError(400, log_message=msg, reason=msg),
                  None))
        return

    # Fake source and receiver to be able to reuse the generic waveform
    # serializer.
    source = ForceSource(latitude=90.0, longitude=90.0,
                         depth_in_m=source_depth_in_m,
                         origin_time=origintime)
    receiver = Receiver(latitude=90.0 - epicentral_distance_degree,
                        longitude=0.0, depth_in_m=0.0)

    _validate_and_write_waveforms(st=st, callback=callback,
                                  starttime=starttime, endtime=endtime,
                                  scale=1.0, source=source, receiver=receiver,
                                  db=db, label=label, format=format)


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
        super(GreensFunctionHandler, self).__init__(*args, **kwargs)

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

        if info.components != "vertical and horizontal":
            msg = ("Database requires vertical AND horizontal components to "
                   "be able to compute Green's functions.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure that epicentral disance and source depth are in reasonable
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
        response, mu = yield tornado.gen.Task(
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

        # Set and thus send the mu header.
        self.set_header("Instaseis-Mu", "%f" % mu)

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

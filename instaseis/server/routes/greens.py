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
from ...base_instaseis_db import _get_seismogram_times
from ..util import run_async, IOQueue, _validtimesetting
from ..instaseis_request import InstaseisRequestHandler
from ...helpers import geocentric_to_wgs84_latitude


@run_async
def _get_greens(db, epicentral_distance_degree, source_depth_in_m, units, dt,
                kernelwidth, origintime, starttime, endtime, format, label,
                callback):
    """
    Extract a seismogram from the passed db and write it either to a MiniSEED
    or a SACZIP file.

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
        st = db.get_greens_seiscomp(
            epicentral_distance_degree=epicentral_distance_degree,
            source_depth_in_m=source_depth_in_m, origin_time=origintime,
            kind=units, return_obspy_stream=True, dt=dt,
            kernelwidth=kernelwidth)
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
    elif format == "sac":
        byte_strings = []
        for tr in st:
            # Write SAC headers.
            tr.stats.sac = obspy.core.AttribDict()
            # Write WGS84 coordinates to the SAC files.
            tr.stats.sac.stla = geocentric_to_wgs84_latitude(
                90.0 - epicentral_distance_degree)
            tr.stats.sac.stlo = 0.0
            tr.stats.sac.stdp = 0.0
            tr.stats.sac.stel = 0.0
            tr.stats.sac.evla = geocentric_to_wgs84_latitude(90.)
            tr.stats.sac.evlo = 90.
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


class GreensHandler(InstaseisRequestHandler):
    # Define the arguments for the Greens endpoint.
    greens_arguments = {
        "units": {"type": str, "default": "displacement"},
        "dt": {"type": float},
        "kernelwidth": {"type": int, "default": 12},
        "label": {"type": str},

        # Source parameters.
        "sourcedistanceindegree": {"type": float, "required": True},
        "sourcedepthinmeters": {"type": float, "required": True},

        # Time parameters.
        "origintime": {"type": obspy.UTCDateTime},
        "starttime": {"type": _validtimesetting,
                      "format": "Datetime String/Float/Phase+-Offset"},
        "endtime": {"type": _validtimesetting,
                    "format": "Datetime String/Float/Phase+-Offset"},

        "format": {"type": str, "default": "saczip"}
    }

    def __init__(self, *args, **kwargs):
        self.__connection_closed = False
        InstaseisRequestHandler.__init__(self, *args, **kwargs)

    def parse_arguments(self):
        # Make sure that no additional arguments are passed.
        unknown_arguments = set(self.request.arguments.keys()).difference(set(
            self.greens_arguments.keys()))
        if unknown_arguments:
            msg = "The following unknown parameters have been passed: %s" % (
                ", ".join("'%s'" % _i for _i in sorted(unknown_arguments)))
            raise tornado.web.HTTPError(400, log_message=msg,
                                        reason=msg)

        # Check for duplicates.
        duplicates = []
        for key, value in self.request.arguments.items():
            if len(value) == 1:
                continue
            else:
                duplicates.append(key)
        if duplicates:
            msg = "Duplicate parameters: %s" % (
                ", ".join("'%s'" % _i for _i in sorted(duplicates)))
            raise tornado.web.HTTPError(400, log_message=msg,
                                        reason=msg)

        args = obspy.core.AttribDict()
        for name, properties in self.greens_arguments.items():
            if "required" in properties:
                try:
                    value = self.get_argument(name)
                except:
                    msg = "Required parameter '%s' not given." % name
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            else:
                if "default" in properties:
                    default = properties["default"]
                else:
                    default = None
                value = self.get_argument(name, default=default)
            if value is not None:
                try:
                    value = properties["type"](value)
                except:
                    if "format" in properties:
                        msg = "Parameter '%s' must be formatted as: '%s'" % (
                            name, properties["format"])
                    else:
                        msg = ("Parameter '%s' could not be converted to "
                               "'%s'.") % (
                            name, str(properties["type"].__name__))
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            setattr(args, name, value)

        # Validate some of them right here.
        self.validate_parameters(args)

        return args

    def on_connection_close(self):
        """
        Called when the client cancels the connection. Then the loop
        requesting seismograms will stop.
        """
        InstaseisRequestHandler.on_connection_close(self)
        self.__connection_closed = True

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

        # Make sure, epicentral disance and source depth are in reasonable
        # ranges
        if args.sourcedistanceindegree is not None and \
                not 0.0 <= args.sourcedistanceindegree <= 180.0:
            msg = ("Epicentral distance should be in [0, 180].")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        if args.sourcedepthinmeters is not None and \
                not 0.0 <= args.sourcedepthinmeters <= info.planet_radius:
            msg = ("Source depth should be in [0, planet radius].")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure the unit arguments is valid.
        args.units = args.units.lower()
        if args.units not in ["displacement", "velocity", "acceleration"]:
            msg = ("Unit must be one of 'displacement', 'velocity', "
                   "or 'acceleration'")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure the output format is valid.
        args.format = args.format.lower()
        if args.format not in ("miniseed", "saczip"):
            msg = ("Format must either be 'miniseed' or 'saczip'.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # If its essentially equal to the internal sampling rate just set it
        # to equal to ease the following comparisons.
        if args.dt and abs(args.dt - info.dt) / args.dt < 1E-7:
            args.dt = info.dt

        # Make sure that dt, if given is larger then 0.01. This should still
        # be plenty for any use case but prevents the server from having to
        # send massive amounts of data in the case of user errors.
        if args.dt is not None and args.dt < 0.01:
            msg = ("The smallest possible dt is 0.01. Please choose a "
                   "smaller value and resample locally if needed.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Also make sure it does not downsample.
        if args.dt is not None and args.dt > info.dt:
            msg = ("Cannot downsample. The sampling interval of the database "
                   "is %.5f seconds. Make sure to choose a smaller or equal "
                   "one." % info.dt)
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure the interpolation kernel width is sensible. Don't allow
        # values smaller than 1 or larger than 20.
        if not (1 <= args.kernelwidth <= 20):
            msg = ("`kernelwidth` must not be smaller than 1 or larger than "
                   "20.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

    def parse_time_settings(self, args):
        """
        Attempt to figure out the time settings.

        This is pretty messy unfortunately. After this method has been
        called, args.origintime will always be set to an absolute time.

        args.starttime and args.endtime will either be set to absolute times
        or dictionaries describing phase relative offsets.

        Returns the minium possible start- and the maximum possible endtime.
        """
        if args.origintime is None:
            args.origintime = obspy.UTCDateTime(0)

        # The origin time will be always set. If the starttime is not set,
        # set it to the origin time.
        if args.starttime is None:
            args.starttime = args.origintime

        # Now it becomes a bit ugly. If the starttime is a float, treat it
        # relative to the origin time.
        if isinstance(args.starttime, float):
            args.starttime = args.origintime + args.starttime

        # Now deal with the endtime.
        if isinstance(args.endtime, float):
            # If the start time is already known as an absolute time,
            # just add it.
            if isinstance(args.starttime, obspy.UTCDateTime):
                args.endtime = args.starttime + args.endtime
            # Otherwise the start time has to be a phase relative time and
            # is dealt with later.
            else:
                assert isinstance(args.starttime, obspy.core.AttribDict)

        # Figure out the maximum temporal range of the seismograms.
        ti = _get_seismogram_times(
            info=self.application.db.info, origin_time=args.origintime,
            dt=args.dt, kernelwidth=args.kernelwidth,
            remove_source_shift=False, reconvolve_stf=False)

        # If the endtime is not set, do it here.
        if args.endtime is None:
            args.endtime = ti["endtime"]

        # Do a couple of sanity checks here.
        if isinstance(args.starttime, obspy.UTCDateTime):
            # The desired seismogram start time must be before the end time of
            # the seismograms.
            if args.starttime >= ti["endtime"]:
                msg = ("The `starttime` must be before the seismogram ends.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            # Arbitrary limit: The starttime can be at max one hour before the
            # origin time.
            if args.starttime < (ti["starttime"] - 3600):
                msg = ("The seismogram can start at the maximum one hour "
                       "before the origin time.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if isinstance(args.endtime, obspy.UTCDateTime):
            # The endtime must be within the seismogram window
            if not (ti["starttime"] <= args.endtime <= ti["endtime"]):
                msg = ("The end time of the seismograms lies outside the "
                       "allowed range.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        return ti["starttime"], ti["endtime"]

    def set_headers(self, args):
        if args.format == "miniseed":
            content_type = "application/octet-stream"
        elif args.format == "saczip":
            content_type = "application/zip"
        self.set_header("Content-Type", content_type)

        FILE_ENDINGS_MAP = {
            "miniseed": "mseed",
            "saczip": "zip"}

        if args.label:
            label = args.label
        else:
            label = "instaseis_greens"

        filename = "%s_%s.%s" % (
            label,
            str(obspy.UTCDateTime()).replace(":", "_"),
            FILE_ENDINGS_MAP[args.format])

        self.set_header("Content-Disposition",
                        "attachment; filename=%s" % filename)

    def get_ttime(self, source, receiver, phase):
        if self.application.travel_time_callback is None:
            msg = "Server does not support travel time calculations."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)
        try:
            tt = self.application.travel_time_callback(
                sourcelatitude=source.latitude,
                sourcelongitude=source.longitude,
                sourcedepthinmeters=source.depth_in_m,
                receiverlatitude=receiver.latitude,
                receiverlongitude=receiver.longitude,
                receiverdepthinmeters=receiver.depth_in_m,
                phase_name=phase)
        except ValueError as e:
            err_msg = str(e)
            if err_msg.lower().startswith("invalid phase name"):
                msg = "Invalid phase name: %s" % phase
            else:
                msg = "Failed to calculate travel time due to: %s" % err_msg
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        return tt

    def validate_geometry(self, source, receiver):
        """
        Validate the source-receiver geometry.
        """
        info = self.application.db.info

        # XXX: Will have to be changed once we have a database recorded for
        # example on the ocean bottom.
        if info.is_reciprocal:
            # Receiver must be at the surface.
            if receiver.depth_in_m is not None:
                if receiver.depth_in_m != 0.0:
                    msg = "Receiver must be at the surface for reciprocal " \
                          "databases."
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            # Source depth must be within the allowed range.
            if not ((info.planet_radius - info.max_radius) <=
                    source.depth_in_m <=
                    (info.planet_radius - info.min_radius)):
                msg = ("Source depth must be within the database range: %.1f "
                       "- %.1f meters.") % (
                        info.planet_radius - info.max_radius,
                        info.planet_radius - info.min_radius)
                raise tornado.web.HTTPError(400, log_message=msg,
                                            reason=msg)
        else:
            # The source depth must coincide with the one in the database.
            if source.depth_in_m != info.source_depth * 1000:
                    msg = "Source depth must be: %.1f km" % info.source_depth
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)

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
        rec_latitude, rec_longitude = 90. - args.sourcedistanceindegree, 0.
        source = Source(src_latitude, src_longitude, args.sourcedepthinmeters)
        receivers = [Receiver(rec_latitude, rec_longitude)]

        # If a zip file is requested, initialize it here and write to custom
        # buffer object.
        if args.format == "saczip":
            buf = IOQueue()
            zip_file = zipfile.ZipFile(buf, mode="w")

        # Count the number of successful extractions. Phase relative offsets
        # could result in no actually calculated seismograms. In that case
        # we would like to raise an error.
        count = 0

        # XXX: only one reqest here, keeping the loop structure to enable the
        # 'count' test as in the seismograms route
        for receiver in receivers:
            # Check if the connection is still open. The __connection_closed
            # flag is set by the on_connection_close() method. This is
            # pretty manual right now. Maybe there is a better way? This
            # enables to server to stop serving if the connection has been
            # cancelled on the client side.
            if self.__connection_closed:
                self.flush()
                self.finish()
                return

            # Check if start- or end time are phase relative. If yes
            # calculate the new start- and/or end time.
            if isinstance(args.starttime, obspy.core.AttribDict):
                tt = self.get_ttime(source=source, receiver=receiver,
                                    phase=args.starttime["phase"])
                if tt is None:
                    continue
                starttime = args.origintime + tt + args.starttime["offset"]
            else:
                starttime = args.starttime

            if starttime < min_starttime - 3600.0:
                continue

            if isinstance(args.endtime, obspy.core.AttribDict):
                tt = self.get_ttime(source=source, receiver=receiver,
                                    phase=args.endtime["phase"])
                if tt is None:
                    continue
                endtime = args.origintime + tt + args.endtime["offset"]
            # Endtime relative to phase relative starttime.
            elif isinstance(args.endtime, float):
                endtime = starttime + args.endtime
            else:
                endtime = args.endtime

            if endtime > max_endtime:
                continue

            # Validate the source-receiver geometry.
            self.validate_geometry(source=source, receiver=receiver)

            # Yield from the task. This enables a context switch and thus
            # async behaviour.
            response = yield tornado.gen.Task(
                _get_greens,
                db=self.application.db,
                epicentral_distance_degree=args.sourcedistanceindegree,
                source_depth_in_m=args.sourcedepthinmeters, units=args.units,
                dt=args.dt, kernelwidth=args.kernelwidth,
                origintime=args.origintime, starttime=starttime,
                endtime=endtime, format=args.format, label=args.label)

            # If an exception is returned from the task, re-raise it here.
            if isinstance(response, Exception):
                raise response
            # It might return a list, in that case each item is a bytestring
            # of SAC file.
            elif isinstance(response, list):
                assert args.format == "saczip"
                for filename, content in response:
                    zip_file.writestr(filename, content)
                for data in buf:
                    self.write(data)
            # Otherwise it contain MiniSEED which can just directly be
            # streamed.
            else:
                self.write(response)
            self.flush()

            count += 1

        # If nothing is written, raise an error. This should really only
        # happen with phase relative offsets with phases not coinciding with
        # the source - receiver geometry.
        if not count:
            msg = ("No seismograms found for the given phase relative "
                   "offsets. This could either be due to the chosen phase "
                   "not existing for the specific source-receiver geometry "
                   "or arriving too late/with too large offsets if the "
                   "database is not long enough.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Write the end of the zipfile in case necessary.
        if args.format == "saczip":
            zip_file.close()
            for data in buf:
                self.write(data)

        self.finish()

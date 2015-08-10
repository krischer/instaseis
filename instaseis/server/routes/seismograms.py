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
import zipfile

import numpy as np
import obspy
import tornado.gen
import tornado.web

from ... import Source, ForceSource, Receiver
from ...base_instaseis_db import _get_seismogram_times
from ..util import run_async
from ..instaseis_request import InstaseisRequestHandler

# Valid phase offset pattern including capture groups.
PHASE_OFFSET_PATTERN = re.compile(r"(^[A-Za-z0-9^]+)([\+-])([\deE\.\-\+]+$)")


@run_async
def _get_seismogram(db, source, receiver, components, units, dt, a_lanczos,
                    starttime, endtime, format, label, callback):
    """
    Extract a seismogram from the passed db and write it either to a MiniSEED
    or a SACZIP file.

    :param db: An open instaseis database.
    :param source: An instaseis source.
    :param receiver: An instaseis receiver.
    :param components: The components.
    :param units: The desired units.
    :param remove_source_shift: Remove the source time shift or not.
    :param dt: dt to resample to.
    :param a_lanczos: Width of the Lanczos kernel.
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
        st = db.get_seismograms(
            source=source, receiver=receiver, components=components,
            kind=units, remove_source_shift=False,
            reconvolve_stf=False, return_obspy_stream=True, dt=dt,
            a_lanczos=a_lanczos)
    except Exception:
        msg = ("Could not extract seismogram. Make sure, the components "
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
            with io.BytesIO() as temp:
                tr.write(temp, format="sac")
                temp.seek(0, 0)
                filename = "%s%s.sac" % (label, tr.id)
                byte_strings.append((filename, temp.read()))
        callback(byte_strings)
    else:
        # Checked above and cannot really happen.
        raise NotImplementedError


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


def _tolist(value, count):
    value = [float(i) for i in value.split(",")]
    if len(value) not in count:
        raise ValueError
    return value


def _momenttensor(value):
    return _tolist(value, (6,))


def _doublecouple(value):
    return _tolist(value, (3, 4))


def _forcesource(value):
    return _tolist(value, (3,))


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


class SeismogramsHandler(InstaseisRequestHandler):
    # Define the arguments for the seismogram endpoint.
    seismogram_arguments = {
        "components": {"type": str, "default": "ZNE"},
        "units": {"type": str, "default": "displacement"},
        "dt": {"type": float},
        "alanczos": {"type": int, "default": 12},
        "label": {"type": str},

        # Source parameters.
        "sourcelatitude": {"type": float},
        "sourcelongitude": {"type": float},
        "sourcedepthinmeters": {"type": float},

        # Source can either be given as the moment tensor components in Nm.
        "sourcemomenttensor": {"type": _momenttensor},
        # Or as strike, dip, rake and M0.
        "sourcedoublecouple": {"type": _doublecouple},
        # Or as a force source.
        "sourceforce": {"type": _forcesource},

        # Or last but not least by specifying an event id.
        "eventid": {"type": str},

        # Time parameters.
        "origintime": {"type": obspy.UTCDateTime},
        "starttime": {"type": _validtimesetting},
        "endtime": {"type": _validtimesetting},

        # Receivers can be specified either directly via their coordinates.
        # In that case one can assign a network and station code.
        "receiverlatitude": {"type": float},
        "receiverlongitude": {"type": float},
        "receiverdepthinmeters": {"type": float, "default": 0.0},
        "networkcode": {"type": str, "default": "XX"},
        "stationcode": {"type": str, "default": "SYN"},

        # Or by querying a database.
        "network": {"type": str},
        "station": {"type": str},

        "format": {"type": str, "default": "saczip"}
    }

    def __init__(self, *args, **kwargs):
        self.__connection_closed = False
        InstaseisRequestHandler.__init__(self, *args, **kwargs)

    def parse_arguments(self):
        # Make sure that no additional arguments are passed.
        unknown_arguments = set(self.request.arguments.keys()).difference(set(
            self.seismogram_arguments.keys()))
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
            elif len(value) == 0:
                # This should not happen.
                raise NotImplementedError
            else:
                duplicates.append(key)
        if duplicates:
            msg = "Duplicate parameters: %s" % (
                ", ".join("'%s'" % _i for _i in sorted(duplicates)))
            raise tornado.web.HTTPError(400, log_message=msg,
                                        reason=msg)

        args = obspy.core.AttribDict()
        for name, properties in self.seismogram_arguments.items():
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
                    msg = "Parameter '%s' could not be converted to '%s'." % (
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

        # Make sure that dt, if given is larger then 0.01. This should still
        # be plenty for any use case but prevents the server from having to
        # send massive amounts of data in the case of user errors.
        if args.dt is not None and args.dt < 0.01:
            msg = ("The smallest possible dt is 0.01. Please choose a "
                   "smaller value and resample locally if needed.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure the lanczos window width is sensible. Don't allow values
        # smaller than 2 or larger than 20.
        if not (1 <= args.alanczos <= 20):
            msg = ("`alanczos` must not be smaller than 1 or larger than 20.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # The networkcode and stationcode parameters have a maximum number
        # of letters.
        if args.stationcode and len(args.stationcode) > 5:
            msg = "'stationcode' must have 5 or fewer letters."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if args.networkcode and len(args.networkcode) > 2:
            msg = "'networkcode' must have 2 or fewer letters."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        all_src_params = set(["sourcemomenttensor", "sourcedoublecouple",
                              "sourceforce", "sourcelatitude",
                              "sourcelongitude", "sourcedepthinmeters"])
        given_params = set([_i for _i in all_src_params
                            if getattr(args, _i) is not None])
        if args.eventid is not None:
            if not self.application.event_info_callback:
                msg = ("Server does not support event information and thus no "
                       "event queries.")
                raise tornado.web.HTTPError(404, log_message=msg, reason=msg)
            # If the event id is given, the origin time cannot be given as
            # well.
            if args.origintime is not None:
                msg = ("'eventid' and 'origintime' parameters cannot both be "
                       "passed at the same time.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

            # If the eventid is given, all the other source parameters must
            # be None.
            if given_params:
                msg = ("The following parameters cannot be used if "
                       "'eventid' is a parameter: %s" % ', '.join(
                        "'%s'" % i for i in sorted(given_params)))
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        # Otherwise the source locations and exactly one of the other values
        # has to set!
        else:
            if not given_params:
                msg = "No source specified"
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            # Needs all of these.
            required_parameters = set(["sourcelatitude", "sourcelongitude",
                                       "sourcedepthinmeters"])
            # And exactly one of these.
            one_off = set(["sourcemomenttensor", "sourcedoublecouple",
                           "sourceforce", "eventid"])

            missing_parameters = required_parameters.difference(given_params)
            if missing_parameters:
                msg = "The following required parameters are missing: %s" % (
                    ", ".join("'%s'" % _i
                              for _i in sorted(missing_parameters)))
                raise tornado.web.HTTPError(
                    400, log_message=msg, reason=msg)

            has_parameters = given_params.intersection(one_off)
            if len(has_parameters) > 1:
                msg = "Only one of these parameters can be given " \
                      "simultaneously: %s" % (
                        ", ".join("'%s'" % _i
                                  for _i in sorted(has_parameters)))
                raise tornado.web.HTTPError(
                    400, log_message=msg, reason=msg)
            elif not has_parameters:
                msg = "One of the following has to be given: %s" % (
                          ", ".join("'%s'" % _i
                                    for _i in sorted(one_off)))
                raise tornado.web.HTTPError(
                    400, log_message=msg, reason=msg)

        # Figure out who the station coordinates are specified.
        direct_receiver_settings = [
            i is not None
            for i in (args.receiverlatitude, args.receiverlongitude)]
        query_receivers = [i is not None for i in (args.network, args.station)]
        if any(direct_receiver_settings) and any(query_receivers):
            msg = ("Receiver coordinates can either be specified by passing "
                   "the coordinates, or by specifying query parameters, "
                   "but not both.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        elif not(all(direct_receiver_settings) or all(query_receivers)):
            msg = ("Must specify a full set of coordinates or a full set of "
                   "receiver parameters.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        elif all(direct_receiver_settings) and all(query_receivers):
            # Should not happen.
            raise NotImplementedError

        # Make sure that the station coordinates callback is available if
        # needed. Otherwise raise a 404.
        if all(query_receivers) and \
                not self.application.station_coordinates_callback:
            msg = ("Server does not support station coordinates and thus no "
                   "station queries.")
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        if len(args.components) > 5:
            msg = "A maximum of 5 components can be requested."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if not args.components:
            msg = "A request with no components will not return anything..."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

    def get_source(self, args, __event):
        # Source can be either directly specified or by passing an event id.
        if args.eventid is not None:
            # Use previously extracted event information.
            source = Source(**__event)
        # Otherwise parse it to one of the supported source types.
        else:
            if args.sourcemomenttensor:
                m = args.sourcemomenttensor
                try:
                    source = Source(latitude=args.sourcelatitude,
                                    longitude=args.sourcelongitude,
                                    depth_in_m=args.sourcedepthinmeters,
                                    m_rr=m[0], m_tt=m[1],
                                    m_pp=m[2], m_rt=m[3],
                                    m_rp=m[4], m_tp=m[5],
                                    origin_time=args.origintime)
                except:
                    msg = ("Could not construct moment tensor source with "
                           "passed parameters. Check parameters for "
                           "sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            elif args.sourcedoublecouple:
                m = args.sourcedoublecouple

                # The seismic moment defaults to 1E19.
                if len(m) == 4:
                    m0 = m[3]
                else:
                    m0 = 1E19

                try:
                    source = Source.from_strike_dip_rake(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        strike=m[0], dip=m[1], rake=m[2],
                        M0=m0, origin_time=args.origintime)
                except:
                    msg = ("Could not construct the source from the "
                           "passed strike/dip/rake parameters. Check "
                           "parameter for sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            elif args.sourceforce:
                m = args.sourceforce
                try:
                    source = ForceSource(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        f_r=m[0], f_t=m[1], f_p=m[2],
                        origin_time=args.origintime)
                except:
                    msg = ("Could not construct force source with passed "
                           "parameters. Check parameters for sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            else:
                msg = "No/insufficient source parameters specified"
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        return source

    def get_receivers(self, args):
        receivers = []

        # Construct either a single receiver object.
        if args.receiverlatitude is not None:
            try:
                receiver = Receiver(latitude=args.receiverlatitude,
                                    longitude=args.receiverlongitude,
                                    network=args.networkcode,
                                    station=args.stationcode,
                                    depth_in_m=args.receiverdepthinmeters)
            except:
                msg = ("Could not construct receiver with passed parameters. "
                       "Check parameters for sanity.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            receivers.append(receiver)
        # Or a list of receivers.
        elif args.network is not None and args.station is not None:
            networks = args.network.split(",")
            stations = args.station.split(",")

            coordinates = self.application.station_coordinates_callback(
                networks=networks, stations=stations)

            if not coordinates:
                msg = "No coordinates found satisfying the query."
                raise tornado.web.HTTPError(
                    404, log_message=msg, reason=msg)
                self.application.station_coordinates_callback(
                    networks=args.network, stations=args.station)

            for station in coordinates:
                try:
                    receivers.append(Receiver(
                        latitude=station["latitude"],
                        longitude=station["longitude"],
                        network=station["network"],
                        station=station["station"],
                        depth_in_m=0))
                except:
                    msg = ("Could not construct receiver with passed "
                           "parameters. Check parameters for sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
        else:
            msg = "Should not happen."
            raise tornado.web.HTTPError(500, log_message=msg, reason=msg)

        return receivers

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
        # Same with the endtime
        if isinstance(args.endtime, float):
            if isinstance(args.starttime, obspy.UTCDateTime):
                args.endtime = args.starttime + args.endtime
            else:
                args.endtime = args.origintime + args.endtime

        # Figure out the maximum temporal range of the seismograms.
        ti = _get_seismogram_times(
            info=self.application.db.info, origin_time=args.origintime,
            dt=args.dt, a_lanczos=args.alanczos, remove_source_shift=False,
            reconvolve_stf=False)

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
            label = "instaseis_seismogram"

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
                msg = "Failed to calculate travel time due to: %s" % str(e)
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        return tt

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        # Parse the arguments. This will also perform a number of sanity
        # checks.
        args = self.parse_arguments()

        if args.eventid is not None:
            # It has to be extracted here to get the origin time which is
            # needed to parse the time settings which might in turn be
            # needed for the sources. This results in a bit of spaghetti code
            # but that's just how it is...
            try:
                __event = self.application.event_info_callback(args.eventid)
            except ValueError:
                msg = "Event not found."
                raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

            if not isinstance(__event, dict) or \
                    sorted(__event.keys()) != sorted((
                    "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp", "latitude",
                    "longitude", "depth_in_m", "origin_time")):
                msg = "Event callback returned an invalid result."
                raise tornado.web.HTTPError(500, log_message=msg, reason="")
            __event["origin_time"] = obspy.UTCDateTime(__event["origin_time"])

            # In case the event is extracted, set the origin time to the
            # time of the event.
            args.origintime = __event["origin_time"]
        else:
            __event = None

        min_starttime, max_endtime = self.parse_time_settings(args)
        self.set_headers(args)

        source = self.get_source(args, __event)

        # Generating even 100'000 receivers only takes ~150ms so its totally
        # ok to generate them all at once here. The time to generate and
        # send the seismograms will dominate.
        receivers = self.get_receivers(args)

        # If a zip file is requested, initialize it here and write to custom
        # buffer object.
        if args.format == "saczip":
            buf = IOQueue()
            zip_file = zipfile.ZipFile(buf, mode="w")

        # Count the number of successful extractions. Phase relative offsets
        # could result in no actually calculated seismograms. In that case
        # we would like to raise an error.
        count = 0

        # Loop over each receiver, get the synthetics and stream it to the
        # user.
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
            else:
                endtime = args.endtime

            if endtime > max_endtime:
                continue

            # Yield from the task. This enables a context switch and thus
            # async behaviour.
            response = yield tornado.gen.Task(
                _get_seismogram,
                db=self.application.db, source=source, receiver=receiver,
                components=list(args.components), units=args.units, dt=args.dt,
                a_lanczos=args.alanczos, starttime=starttime,
                endtime=endtime, format=args.format,
                label=args.label)

            # If an exception is returned from the task, re-raise it here.
            if isinstance(response, Exception):
                raise response
            # It might return a list, in that case each item is a bytestring
            # of SAC file.
            elif isinstance(response, list):
                if args.format != "saczip":
                    raise NotImplemented
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
            msg = "No seismograms could be calculated matching the query."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Write the end of the zipfile in case necessary.
        if args.format == "saczip":
            zip_file.close()
            for data in buf:
                self.write(data)

        self.finish()

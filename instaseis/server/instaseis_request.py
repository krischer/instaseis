#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Instaseis Request handler currently only settings default headers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import obspy
import tornado
from ..database_interfaces.base_instaseis_db import _get_seismogram_times
from .. import Receiver, FiniteSource

from .. import __version__


class InstaseisRequestHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Server", "InstaseisServer/%s" % __version__)


class InstaseisTimeSeriesHandler(with_metaclass(ABCMeta,
                                                InstaseisRequestHandler)):
    arguments = None
    connection_closed = False
    default_label = ""
    default_origin_time = obspy.UTCDateTime(0)

    def __init__(self, *args, **kwargs):
        super(InstaseisTimeSeriesHandler, self).__init__(*args, **kwargs)

    def on_connection_close(self):  # pragma: no cover
        """
        Called when the client cancels the connection. Then the loop
        requesting seismograms will stop.
        """
        InstaseisRequestHandler.on_connection_close(self)
        self.__connection_closed = True

    def parse_arguments(self):
        # Make sure that no additional arguments are passed.
        unknown_arguments = set(self.request.arguments.keys()).difference(set(
            self.arguments.keys()))

        # Remove the body arguments as they don't count.
        unknown_arguments = unknown_arguments.difference(set(
            self.request.body_arguments.keys()))

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
        for name, properties in self.arguments.items():
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
        self.validate_common_parameters(args)
        self.validate_parameters(args)

        return args

    def validate_common_parameters(self, args):
        """
        Also ensures some consistency across the routes.
        """
        info = self.application.db.info

        if "components" in self.arguments:
            if len(args.components) > 5:
                msg = "A maximum of 5 components can be requested."
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

            if not args.components:
                msg = ("A request with no components will not return "
                       "anything...")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure the unit arguments is valid.
        if "units" in self.arguments:
            args.units = args.units.lower()
            if args.units not in ["displacement", "velocity", "acceleration"]:
                msg = ("Unit must be one of 'displacement', 'velocity', "
                       "or 'acceleration'")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure the output format is valid.
        if "format" in self.arguments:
            args.format = args.format.lower()
            if args.format not in ("miniseed", "saczip"):
                msg = ("Format must either be 'miniseed' or 'saczip'.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # If its essentially equal to the internal sampling rate just set it
        # to equal to ease the following comparisons.
        if "dt" in self.arguments:
            if args.dt and abs(args.dt - info.dt) / args.dt < 1E-7:
                args.dt = info.dt

            # Make sure that dt, if given is larger then 0.01. This should
            # still be plenty for any use case but prevents the server from
            # having to send massive amounts of data in the case of user
            # errors.
            if args.dt is not None and args.dt < 0.01:
                msg = ("The smallest possible dt is 0.01. Please choose a "
                       "smaller value and resample locally if needed.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

            # Also make sure it does not downsample.
            if args.dt is not None and args.dt > info.dt:
                msg = ("Cannot downsample. The sampling interval of the "
                       "database is %.5f seconds. Make sure to choose a "
                       "smaller or equal one." % info.dt)
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if "kernelwidth" in self.arguments:
            # Make sure the interpolation kernel width is sensible. Don't allow
            # values smaller than 1 or larger than 20.
            if not (1 <= args.kernelwidth <= 20):
                msg = ("`kernelwidth` must not be smaller than 1 or larger "
                       "than 20.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

    @abstractmethod
    def validate_parameters(self, args):
        """
        Implement this function to make checks not already performed in
        validate_common_parameters().
        """
        raise NotImplementedError

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
            args.origintime = self.default_origin_time

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
        if "format" not in args:
            format = "miniseed"
        else:
            format = args.format

        if format == "miniseed":
            content_type = "application/vnd.fdsn.mseed"
        elif format == "saczip":
            content_type = "application/zip"
        self.set_header("Content-Type", content_type)

        FILE_ENDINGS_MAP = {
            "miniseed": "mseed",
            "saczip": "zip"}

        if "label" in args and args.label:
            label = args.label
        else:
            label = self.default_label

        filename = "%s_%s.%s" % (
            label,
            str(obspy.UTCDateTime()).replace(":", "_"),
            FILE_ENDINGS_MAP[format])

        self.set_header("Content-Disposition",
                        "attachment; filename=%s" % filename)

    def get_ttime(self, source, receiver, phase):
        if self.application.travel_time_callback is None:
            msg = "Server does not support travel time calculations."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        # Finite sources will perform these calculations with the hypocenter.
        if isinstance(source, FiniteSource):
            src_latitude = source.hypocenter_latitude
            src_longitude = source.hypocenter_longitude
            src_depth_in_m = source.hypocenter_depth_in_m
        # or any single source.
        else:
            src_latitude = source.latitude
            src_longitude = source.longitude
            src_depth_in_m = source.depth_in_m

        try:
            tt = self.application.travel_time_callback(
                sourcelatitude=src_latitude,
                sourcelongitude=src_longitude,
                sourcedepthinmeters=src_depth_in_m,
                receiverlatitude=receiver.latitude,
                receiverlongitude=receiver.longitude,
                receiverdepthinmeters=receiver.depth_in_m,
                phase_name=phase,
                db_info=self.application.db.info)
        except ValueError as e:
            err_msg = str(e)
            if err_msg.lower().startswith("invalid phase name"):
                msg = "Invalid phase name: %s" % phase
            # This is just a safeguard - its save to not coverage test it.
            else:  # pragma: no cover
                msg = "Failed to calculate travel time due to: %s" % err_msg
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        return tt

    def validate_geometry(self, source, receiver):
        """
        Validate the source-receiver geometry.
        """
        info = self.application.db.info

        # Any single...
        if hasattr(source, "latitude"):
            src_depth_in_m = source.depth_in_m
        # Or a finite source...
        else:
            src_depth_in_m = source.hypocenter_depth_in_m

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
                    src_depth_in_m <=
                    (info.planet_radius - info.min_radius)):
                msg = ("Source depth must be within the database range: %.1f "
                       "- %.1f meters.") % (
                        info.planet_radius - info.max_radius,
                        info.planet_radius - info.min_radius)
                raise tornado.web.HTTPError(400, log_message=msg,
                                            reason=msg)
        else:
            # The source depth must coincide with the one in the database.
            if src_depth_in_m != info.source_depth * 1000:
                    msg = "Source depth must be: %.1f km" % info.source_depth
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)

    def get_phase_relative_times(self, args, source, receiver, min_starttime,
                                 max_endtime):
        """
        Helper function getting the times for each receiver for
        phase-relative offsets.

        Returns None in case there either is no phase at the
        requested distance or it arrives too late, early for other settings.
        """
        if isinstance(args.starttime, obspy.core.AttribDict):
            tt = self.get_ttime(source=source, receiver=receiver,
                                phase=args.starttime["phase"])
            if tt is None:
                return
            starttime = args.origintime + tt + args.starttime["offset"]
        else:
            starttime = args.starttime

        if starttime < min_starttime - 3600.0:
            return

        if isinstance(args.endtime, obspy.core.AttribDict):
            tt = self.get_ttime(source=source, receiver=receiver,
                                phase=args.endtime["phase"])
            if tt is None:
                return
            endtime = args.origintime + tt + args.endtime["offset"]
        # Endtime relative to phase relative starttime.
        elif isinstance(args.endtime, float):
            endtime = starttime + args.endtime
        else:
            endtime = args.endtime

        if endtime > max_endtime:
            return

        return starttime, endtime

    def get_receivers(self, args):
        # Already checked before - just make sure the settings are valid.
        assert (args.receiverlatitude is not None and
                args.receiverlongitude is not None) or \
               (args.network and args.station)

        receivers = []

        rec_depth = args.receiverdepthinmeters

        # Construct either a single receiver object.
        if args.receiverlatitude is not None:
            try:
                receiver = Receiver(latitude=args.receiverlatitude,
                                    longitude=args.receiverlongitude,
                                    network=args.networkcode,
                                    station=args.stationcode,
                                    location=args.locationcode,
                                    depth_in_m=rec_depth)
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

            for station in coordinates:
                try:
                    receivers.append(Receiver(
                        latitude=station["latitude"],
                        longitude=station["longitude"],
                        network=station["network"],
                        station=station["station"],
                        depth_in_m=0))
                except:
                    msg = ("Station coordinate query returned invalid "
                           "coordinates.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
        return receivers

    def validate_receiver_parameters(self, args):
        """
        Useful for routes that use single receivers.
        """
        # The networkcode and stationcode parameters have a maximum number
        # of letters.
        if args.stationcode and len(args.stationcode) > 5:
            msg = "'stationcode' must have 5 or fewer letters."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if args.networkcode and len(args.networkcode) > 2:
            msg = "'networkcode' must have 2 or fewer letters."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # The location code as well.
        if args.locationcode and len(args.locationcode) > 2:
            msg = "'locationcode' must have 2 or fewer letters."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

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

        # Should not happen.
        assert not (all(direct_receiver_settings) and all(query_receivers))

        # Make sure that the station coordinates callback is available if
        # needed. Otherwise raise a 404.
        if all(query_receivers) and \
                not self.application.station_coordinates_callback:
            msg = ("Server does not support station coordinates and thus no "
                   "station queries.")
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

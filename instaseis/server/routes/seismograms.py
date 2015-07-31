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
import zipfile

import numpy as np
import obspy
import tornado.gen
import tornado.web

from ... import Source, ForceSource, Receiver
from ...helpers import get_band_code
from ...lanczos import interpolate_trace
from ..util import run_async
from ..instaseis_request import InstaseisRequestHandler


@run_async
def _get_seismogram(db, source, receiver, components, unit, dt, a_lanczos,
                    origin_time, starttime, endtime, src_shift, format,
                    callback):
    """
    Extract a seismogram from the passed db and write it either to a MiniSEED
    or a SACZIP file.

    :param db: An open instaseis database.
    :param source: An instaseis source.
    :param receiver: An instaseis receiver.
    :param components: The components.
    :param unit: The desired unit.
    :param remove_source_shift: Remove the source time shift or not.
    :param dt: dt to resample to.
    :param a_lanczos: Width of the Lanczos kernel.
    :param origin_time: The peak of the source time function will be set to
        that.
    :param starttime: The desired start time of the seismogram.
    :param endtime: The desired end time of the seismogram.
    :param src_shift: The peak of the source time function in seconds
        relative to the first sample.
    :param format:
    :param callback: callback function of the coroutine.
    """
    try:
        st = db.get_seismograms(
            source=source, receiver=receiver, components=components,
            kind=unit, remove_source_shift=False,
            reconvolve_stf=False, return_obspy_stream=True, dt=None)
    except Exception:
        msg = ("Could not extract seismogram. Make sure, the components "
               "are valid, and the depth settings are correct.")
        callback(tornado.web.HTTPError(400, log_message=msg, reason=msg))
        return

    for tr in st:
        # Adjust for the source shift.
        tr.stats.starttime = origin_time - src_shift

    # Trim, potentially pad with zeroes. Previous checks ensure that no
    # padding will happen at the end.
    st.trim(starttime, endtime, pad=True, fill_value=0.0, nearest_sample=False)

    for tr in st:
        # Resample now to deal with the padding and what not.
        if dt is not None:
            interpolate_trace(tr, sampling_rate=1.0 / dt, a=a_lanczos)
            # The channel mapping has to be reapplied.
            tr.stats.channel = get_band_code(dt) + tr.stats.channel[1:]

        # Half the filesize but definitely sufficiently accurate.
        tr.data = np.require(tr.data, dtype=np.float32)

    if format == "mseed":
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
                filename = "%s.sac" % tr.id
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


class SeismogramsHandler(InstaseisRequestHandler):
    # Define the arguments for the seismogram endpoint.
    seismogram_arguments = {
        "components": {"type": str, "default": "ZNE"},
        "unit": {"type": str, "default": "displacement"},
        "dt": {"type": float},
        "alanczos": {"type": int, "default": 5},

        # Source parameters.
        "sourcelatitude": {"type": float, "required": True},
        "sourcelongitude": {"type": float, "required": True},
        "sourcedepthinm": {"type": float, "default": 0.0},

        # Source can either be given as the moment tensor components in Nm.
        "mrr": {"type": float},
        "mtt": {"type": float},
        "mpp": {"type": float},
        "mrt": {"type": float},
        "mrp": {"type": float},
        "mtp": {"type": float},

        # Or as strike, dip, rake and M0.
        "strike": {"type": float},
        "dip": {"type": float},
        "rake": {"type": float},
        "M0": {"type": float},

        # Or as a force source.
        "fr": {"type": float},
        "ft": {"type": float},
        "fp": {"type": float},

        # 5 parameters influence the final times of the returned seismograms.
        "origintime": {"type": obspy.UTCDateTime},
        "starttime": {"type": obspy.UTCDateTime},
        "endtime": {"type": obspy.UTCDateTime},
        "duration": {"type": float},

        # Receivers can be specified either directly via their coordinates.
        # In that case one can assign a network and station code.
        "receiverlatitude": {"type": float},
        "receiverlongitude": {"type": float},
        "receiverdepthinm": {"type": float, "default": 0.0},
        "networkcode": {"type": str},
        "stationcode": {"type": str},

        # Or by querying a database.
        "network": {"type": str},
        "station": {"type": str},

        "format": {"type": str, "default": "mseed"}
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
        return args

    def on_connection_close(self):
        """
        Called when the client cancels the connection. Then the loop
        requesting seismograms will stop.
        """
        InstaseisRequestHandler.on_connection_close(self)
        self.__connection_closed = True

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        args = self.parse_arguments()

        # Make sure the unit arguments is valid.
        args.unit = args.unit.lower()
        if args.unit not in ["displacement", "velocity", "acceleration"]:
            msg = ("Unit must be one of 'displacement', 'velocity', "
                   "or 'acceleration'")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Make sure the output format is valid.
        args.format = args.format.lower()
        if args.format not in ("mseed", "saczip"):
            msg = ("Format must either be 'mseed' or 'saczip'.")
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
        if not (2 <= args.alanczos <= 20):
            msg = ("`alanczos` must not be smaller than 2 or larger than 20.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Figure out who the station coordinates are specified.
        direct_receiver_settings = [args.receiverlatitude,
                                    args.receiverlongitude]
        query_receivers = [args.network, args.station]
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

        # Figure out the type of source and construct the source object.
        src_params = {
            "moment_tensor": set(["mrr", "mtt", "mpp", "mrt", "mrp",
                                  "mtp"]),
            "strike_dip_rake": set(["strike", "dip", "rake", "M0"]),
            "force_source": set(["fr", "ft", "fp"])
        }

        if len(args.components) > 5:
            msg = "A maximum of 5 components can be requested."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if not args.components:
            msg = "A request with no components will not return anything..."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Figure out the time settings.
        info = self.application.db.info

        # The time shift necessary to set the origin time to the peak of the
        # source time function. Will be the time of the peak of the original
        # source time function in AxiSEM or the peak of the gaussian
        # function if reconvolved with it.
        src_shift = info.src_shift_samples * info.dt

        # Start time and origin time. If either is not set, one will be set
        # to the other. If neither is set, both will be set to posix timestamp
        # 0.
        if args.origintime is None and args.starttime is None:
            args.origintime = obspy.UTCDateTime(0)
            args.starttime = obspy.UTCDateTime(0)
        elif args.origintime is None:
            args.origintime = args.starttime
        elif args.starttime is None:
            args.starttime = args.origintime

        # Duration and endtime parameters are mutually exclusive.
        if args.duration is not None and args.endtime is not None:
            msg = ("'duration' and 'endtime' parameters cannot both be passed "
                   "at the same time.")
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)
        # Get the temporal extents of the extracted seismograms.
        seismogram_starttime = args.origintime - src_shift
        seismogram_endtime = \
            seismogram_starttime + (info.npts - 1) * info.dt

        # Get the desired endtime.
        if args.duration is None and args.endtime is None:
            args.endtime = seismogram_endtime
        elif args.endtime is None:
            args.endtime = args.starttime + args.duration

        # The desired seismogram start time must be before the end time of the
        # seismograms.
        if args.starttime >= seismogram_endtime:
            msg = ("The `starttime` must be before the seismogram ends.")
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        # The endtime must be within the seismogram window
        if not (seismogram_starttime <= args.endtime <= seismogram_endtime):
            msg = ("The end time of the seismograms lies outside the allowed "
                   "range.")
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        if args.starttime >= args.endtime:
            msg = ("The calculated start time of the seismograms must be "
                   "before the calculated end time.")
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        # Arbitrary limit: The starttime can be at max one hour before the
        # origin time.
        if args.starttime < (seismogram_starttime - 3600):
            msg = ("The seismogram can start at the maximum one hour before "
                   "the origin time.")
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        components = list(args.components)
        for src_type, params in src_params.items():
            src_params = [getattr(args, _i) for _i in params]
            if None in src_params:
                continue
            elif src_type == "moment_tensor":
                try:
                    source = Source(latitude=args.sourcelatitude,
                                    longitude=args.sourcelongitude,
                                    depth_in_m=args.sourcedepthinm,
                                    m_rr=args.mrr, m_tt=args.mtt,
                                    m_pp=args.mpp, m_rt=args.mrt,
                                    m_rp=args.mrp, m_tp=args.mtp)
                except:
                    msg = ("Could not construct moment tensor source with "
                           "passed parameters. Check parameters for sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
                break
            elif src_type == "strike_dip_rake":
                try:
                    source = Source.from_strike_dip_rake(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinm,
                        strike=args.strike, dip=args.dip, rake=args.rake,
                        M0=args.M0)
                except:
                    msg = ("Could not construct the source from the passed "
                           "strike/dip/rake parameters. Check parameter for "
                           "sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
                break
            elif src_type == "force_source":
                try:
                    source = ForceSource(latitude=args.sourcelatitude,
                                         longitude=args.sourcelongitude,
                                         depth_in_m=args.sourcedepthinm,
                                         f_r=args.fr, f_t=args.ft,
                                         f_p=args.fp)
                except:
                    msg = ("Could not construct force source with passed "
                           "parameters. Check parameters for sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
                break
            else:
                # Cannot really happen.
                raise NotImplementedError
        else:
            msg = "No/insufficient source parameters specified"
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if args.format == "mseed":
            content_type = "application/octet-stream"
        elif args.format == "saczip":
            content_type = "application/zip"
        self.set_header("Content-Type", content_type)

        # Generating even 100'000 receivers only takes ~150ms so its totally
        # ok to generate them all at once here. The time to generate and
        # send the seismograms will dominate.

        receivers = []

        # Construct either a single receiver object.
        if all(direct_receiver_settings):
            try:
                receiver = Receiver(latitude=args.receiverlatitude,
                                    longitude=args.receiverlongitude,
                                    network=args.networkcode,
                                    station=args.stationcode,
                                    depth_in_m=args.receiverdepthinm)
            except:
                msg = ("Could not construct receiver with passed parameters. "
                       "Check parameters for sanity.")
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            receivers.append(receiver)
        # Or a list of receivers.
        elif all(query_receivers):
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

        if args.format == "saczip":
            buf = IOQueue()
            zip_file = zipfile.ZipFile(buf, mode="w")

        # For each, get the synthetics, and stream it to the user.
        for receiver in receivers:
            # Check if the connection is still open. The __connection_closed
            # flag is set by the on_connection_close() method. This is
            # pretty manual right now. Maybe there is a better way?
            if self.__connection_closed:
                self.flush()
                self.finish()
                return
            # Yield from the task. This enables a context switch and thus
            # async behaviour.
            response = yield tornado.gen.Task(
                _get_seismogram,
                db=self.application.db, source=source, receiver=receiver,
                components=components, unit=args.unit, dt=args.dt,
                a_lanczos=args.alanczos, origin_time=args.origintime,
                starttime=args.starttime, endtime=args.endtime,
                src_shift=src_shift, format=args.format)

            if isinstance(response, Exception):
                raise response
            elif isinstance(response, list):
                if args.format != "saczip":
                    raise NotImplemented
                for filename, content in response:
                    zip_file.writestr(filename, content)
                for data in buf:
                    self.write(data)
            else:
                self.write(response)
            self.flush()

        if args.format == "saczip":
            # Write the end of the zipfile.
            zip_file.close()
            for data in buf:
                self.write(data)

        FILE_ENDINGS_MAP = {
            "mseed": "mseed",
            "saczip": "zip"}

        filename = "instaseis_seismogram_%s.%s" % (
            str(obspy.UTCDateTime()).replace(":", "_"),
            FILE_ENDINGS_MAP[args.format])

        self.set_header("Content-Disposition",
                        "attachment; filename=%s" % filename)
        self.finish()

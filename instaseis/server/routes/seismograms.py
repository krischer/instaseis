#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import concurrent.futures
import inspect
import io
import json
import os
import re
import zipfile

from jsonschema import validate as json_validate
from jsonschema import ValidationError as JSONValidationError
import numpy as np
import obspy
from obspy.signal.interpolation import lanczos_interpolation
import tornado.gen
import tornado.web

from ... import Source, ForceSource, Receiver
from ..util import (
    IOQueue,
    _validtimesetting,
    _validate_and_write_waveforms,
    get_gaussian_source_time_function,
)
from ..instaseis_request import InstaseisTimeSeriesHandler


executor = concurrent.futures.ThreadPoolExecutor(12)


# Load the JSON schema once.
DATA = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
    ),
    "data",
)
with io.open(os.path.join(DATA, "finite_source_schema.json"), "rt") as fh:
    _json_schema = json.load(fh)


def _get_seismogram(
    db,
    source,
    receiver,
    components,
    units,
    dt,
    kernelwidth,
    starttime,
    endtime,
    scale,
    format,
    label,
):
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
    :param kernelwidth: Width of the interpolation kernel.
    :param starttime: The desired start time of the seismogram.
    :param endtime: The desired end time of the seismogram.
    :param scale: A scalar factor which the seismograms will be multiplied
        with.
    :param format: The output format. Either "miniseed" or "saczip".
    :param label: Prefix for the filename within the SAC zip file.
    """
    if source.sliprate is not None:
        reconvolve_stf = True
    else:
        reconvolve_stf = False

    try:
        st = db.get_seismograms(
            source=source,
            receiver=receiver,
            components=components,
            kind=units,
            remove_source_shift=False,
            reconvolve_stf=reconvolve_stf,
            return_obspy_stream=True,
            dt=dt,
            kernelwidth=kernelwidth,
        )
    except Exception:
        msg = (
            "Could not extract seismogram. Make sure, the components "
            "are valid, and the depth settings are correct."
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg), None

    return _validate_and_write_waveforms(
        st=st,
        starttime=starttime,
        endtime=endtime,
        scale=scale,
        source=source,
        receiver=receiver,
        db=db,
        label=label,
        format=format,
    )


def _parse_validate_and_resample_stf(request, db_info):
    """
    Parses the JSON based STF, validates it, and resamples it.

    :param request: The request.
    :param db_info: Information about the current database.
    """
    if not request.body:
        msg = (
            "The source time function must be given in the body of the "
            "POST request."
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    # Try to parse it as a JSON file.
    with io.BytesIO(request.body) as buf:
        try:
            j = json.loads(buf.read().decode())
        except Exception:
            msg = "The body of the POST request is not a valid JSON file."
            return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    # Validate it.
    try:
        json_validate(j, _json_schema)
    except JSONValidationError as e:
        # Replace the u'' unicode string specifier for consistent error
        # messages.
        msg = "Validation Error in JSON file: " + re.sub(r"u'", "'", e.message)
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    # Make sure the sampling rate is ok.
    if j["sample_spacing_in_sec"] < db_info.dt:
        msg = (
            "'sample_spacing_in_sec' in the JSON file must not be smaller "
            "than the database dt [%.3f seconds]." % db_info.dt
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    # Convert to numpy array.
    j["data"] = np.array(j["data"], np.float64)

    # A couple more custom validations.
    message = None

    # Make sure its not all zeros.
    if np.abs(j["data"]).max() < 1e-20:
        message = (
            "All zero (or nearly all zero) source time functions don't "
            "make any sense."
        )

    # The data must begin and end with zero. The user is responsible for the
    # tapering.
    if j["data"][0] != 0.0 or j["data"][-1] != 0.0:
        message = "Must begin and end with zero."

    if message:
        msg = "STF data did not validate: %s" % message
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    missing_length = (
        db_info.length - (len(j["data"]) - 1) * j["sample_spacing_in_sec"]
    )
    missing_samples = max(
        int(missing_length / j["sample_spacing_in_sec"]) + 1, 0
    )

    # Add a buffer of 20 samples at the beginning and at the end.
    data = np.concatenate(
        [np.zeros(20), j["data"], np.zeros(missing_samples + 20)]
    )

    # Resample it using sinc reconstruction.
    data = lanczos_interpolation(
        data,
        # Account for the additional samples at the beginning.
        old_start=-20 * j["sample_spacing_in_sec"],
        old_dt=j["sample_spacing_in_sec"],
        new_start=0.0,
        new_dt=db_info.dt,
        new_npts=db_info.npts,
        # The large a is okay because we add zeros at the beginning and the
        # end.
        a=12,
        window="blackman",
    )

    # There is potentially some numerical noise on the first sample.
    assert data[0] < 1e-10 * np.abs(data.ptp())
    data[0] = 0.0

    # Normalize the integral to one.
    data /= np.trapz(np.abs(data), dx=db_info.dt)
    j["data"] = data

    return j


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


class SeismogramsHandler(InstaseisTimeSeriesHandler):
    # Define the arguments for the seismogram endpoint.
    arguments = {
        # Default arguments are either 'ZNE', 'Z', or 'NE', depending on
        # what the database supports. Default argument will be set later when
        # the database is known.
        "components": {"type": str},
        "units": {"type": str, "default": "displacement"},
        "dt": {"type": float},
        "kernelwidth": {"type": int, "default": 12},
        "label": {"type": str},
        # Source parameters.
        "sourcelatitude": {"type": float},
        "sourcelongitude": {"type": float},
        "sourcedepthinmeters": {"type": float},
        # Source can either be given as the moment tensor components in Nm.
        "sourcemomenttensor": {
            "type": _momenttensor,
            "format": "Mrr,Mtt,Mpp,Mrt,Mrp,Mtp",
        },
        # Or as strike, dip, rake and M0.
        "sourcedoublecouple": {
            "type": _doublecouple,
            "format": "strike,dip,rake[,M0]",
        },
        # Or as a force source.
        "sourceforce": {"type": _forcesource, "format": "Fr,Ft,Fp"},
        # Scale parameter.
        "scale": {"type": float, "default": 1.0},
        # Source width in seconds. STF will be a gaussian.
        "sourcewidth": {"type": float},
        # Or last but not least by specifying an event id.
        "eventid": {"type": str},
        # Time parameters.
        "origintime": {"type": obspy.UTCDateTime},
        "starttime": {
            "type": _validtimesetting,
            "format": "Datetime String/Float/Phase+-Offset",
        },
        "endtime": {
            "type": _validtimesetting,
            "format": "Datetime String/Float/Phase+-Offset",
        },
        # Receivers can be specified either directly via their coordinates.
        # In that case one can assign a network and station code.
        "receiverlatitude": {"type": float},
        "receiverlongitude": {"type": float},
        "receiverdepthinmeters": {"type": float, "default": 0.0},
        "networkcode": {"type": str, "default": "XX"},
        "stationcode": {"type": str, "default": "SYN"},
        "locationcode": {"type": str, "default": "SE"},
        # Or by querying a database.
        "network": {"type": str},
        "station": {"type": str},
        "format": {"type": str, "default": "saczip"},
    }

    default_label = "instaseis_seismogram"
    # Done here as the time parsing is fairly complex and cannot be done
    # with normal default values.
    default_origin_time = obspy.UTCDateTime(1900, 1, 1)

    def __init__(self, *args, **kwargs):
        super(SeismogramsHandler, self).__init__(*args, **kwargs)
        # Set the correct default arguments.
        self.arguments["components"]["default"] = "".join(
            self.application.db.default_components
        )

    def validate_parameters(self, args):
        """
        Function attempting to validate that the passed parameters are
        valid. Does not need to check the types as that has already been done.
        """
        if args.scale == 0.0:
            msg = (
                "A scale of zero means all seismograms have an amplitude "
                "of zero. No need to get it in the first place."
            )
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if args.sourcewidth is not None:
            if args.sourcewidth < self.application.db.info.period:
                msg = (
                    "The sourcewidth must not be smaller than the mesh "
                    "period of the database (%.3f seconds)."
                    % self.application.db.info.period
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            # Set some reasonable upper limit to stabilize the logic and
            # calculations.
            if args.sourcewidth > 600.0:
                msg = "The sourcewidth must not be larger than 600 seconds."
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        self.validate_receiver_parameters(args)
        self.validate_source_parameters(args)

    def validate_source_parameters(self, args):
        all_src_params = set(
            [
                "sourcemomenttensor",
                "sourcedoublecouple",
                "sourceforce",
                "sourcelatitude",
                "sourcelongitude",
                "sourcedepthinmeters",
            ]
        )
        given_params = set(
            [_i for _i in all_src_params if getattr(args, _i) is not None]
        )
        if args.eventid is not None:
            if not self.application.event_info_callback:
                msg = (
                    "Server does not support event information and thus no "
                    "event queries."
                )
                raise tornado.web.HTTPError(404, log_message=msg, reason=msg)
            # If the event id is given, the origin time cannot be given as
            # well.
            if args.origintime is not None:
                msg = (
                    "'eventid' and 'origintime' parameters cannot both be "
                    "passed at the same time."
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

            # If the eventid is given, all the other source parameters must
            # be None.
            if given_params:
                msg = (
                    "The following parameters cannot be used if "
                    "'eventid' is a parameter: %s"
                    % ", ".join("'%s'" % i for i in sorted(given_params))
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
        # Otherwise the source locations and exactly one of the other values
        # has to set!
        else:
            if not given_params:
                msg = "No source specified"
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            # Needs all of these.
            required_parameters = set(
                ["sourcelatitude", "sourcelongitude", "sourcedepthinmeters"]
            )
            # And exactly one of these.
            one_off = set(
                [
                    "sourcemomenttensor",
                    "sourcedoublecouple",
                    "sourceforce",
                    "eventid",
                ]
            )

            missing_parameters = required_parameters.difference(given_params)
            if missing_parameters:
                msg = "The following required parameters are missing: %s" % (
                    ", ".join("'%s'" % _i for _i in sorted(missing_parameters))
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

            has_parameters = given_params.intersection(one_off)
            if len(has_parameters) > 1:
                msg = (
                    "Only one of these parameters can be given "
                    "simultaneously: %s"
                    % (", ".join("'%s'" % _i for _i in sorted(has_parameters)))
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            elif not has_parameters:
                msg = "One of the following has to be given: %s" % (
                    ", ".join("'%s'" % _i for _i in sorted(one_off))
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

    def get_source(self, args, __event, custom_stf=None):
        # Source can be either directly specified or by passing an event id.
        if args.eventid is not None:
            # Use previously extracted event information.
            source = Source(**__event)
        # Otherwise parse it to one of the supported source types.
        else:
            # Already checked before - just make sure.
            assert (
                args.sourcemomenttensor
                or args.sourcedoublecouple
                or args.sourceforce
            )

            if args.sourcemomenttensor:
                m = args.sourcemomenttensor
                try:
                    source = Source(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        m_rr=m[0],
                        m_tt=m[1],
                        m_pp=m[2],
                        m_rt=m[3],
                        m_rp=m[4],
                        m_tp=m[5],
                        origin_time=args.origintime,
                    )
                except Exception:
                    msg = (
                        "Could not construct moment tensor source with "
                        "passed parameters. Check parameters for "
                        "sanity."
                    )
                    raise tornado.web.HTTPError(
                        400, log_message=msg, reason=msg
                    )
            elif args.sourcedoublecouple:
                m = args.sourcedoublecouple

                # The seismic moment defaults to 1E19.
                if len(m) == 4:
                    m0 = m[3]
                else:
                    m0 = 1e19

                if m0 < 0:
                    msg = "Seismic moment must not be negative."
                    raise tornado.web.HTTPError(
                        400, log_message=msg, reason=msg
                    )

                try:
                    source = Source.from_strike_dip_rake(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        strike=m[0],
                        dip=m[1],
                        rake=m[2],
                        M0=m0,
                        origin_time=args.origintime,
                    )
                except Exception:
                    msg = (
                        "Could not construct the source from the "
                        "passed strike/dip/rake parameters. Check "
                        "parameter for sanity."
                    )
                    raise tornado.web.HTTPError(
                        400, log_message=msg, reason=msg
                    )
            elif args.sourceforce:
                m = args.sourceforce
                try:
                    source = ForceSource(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        f_r=m[0],
                        f_t=m[1],
                        f_p=m[2],
                        origin_time=args.origintime,
                    )
                except Exception:
                    msg = (
                        "Could not construct force source with passed "
                        "parameters. Check parameters for sanity."
                    )
                    raise tornado.web.HTTPError(
                        400, log_message=msg, reason=msg
                    )

        # Add the resampled custom STF to the source object.
        if custom_stf:
            source.sliprate = custom_stf["data"]
            source.dt = self.application.db.info.dt
            source.time_shift = -custom_stf["relative_origin_time_in_sec"]

        return source

    def get_receivers(self, args):
        # Already checked before - just make sure the settings are valid.
        assert (
            args.receiverlatitude is not None
            and args.receiverlongitude is not None
        ) or (args.network and args.station)

        receivers = []

        # Construct either a single receiver object.
        if args.receiverlatitude is not None:
            try:
                receiver = Receiver(
                    latitude=args.receiverlatitude,
                    longitude=args.receiverlongitude,
                    network=args.networkcode,
                    station=args.stationcode,
                    location=args.locationcode,
                    depth_in_m=args.receiverdepthinmeters,
                )
            except Exception:
                msg = (
                    "Could not construct receiver with passed parameters. "
                    "Check parameters for sanity."
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            receivers.append(receiver)
        # Or a list of receivers.
        elif args.network is not None and args.station is not None:
            networks = args.network.split(",")
            stations = args.station.split(",")

            coordinates = self.application.station_coordinates_callback(
                networks=networks, stations=stations
            )

            if not coordinates:
                msg = "No coordinates found satisfying the query."
                raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

            for station in coordinates:
                try:
                    receivers.append(
                        Receiver(
                            latitude=station["latitude"],
                            longitude=station["longitude"],
                            network=station["network"],
                            station=station["station"],
                            depth_in_m=0,
                        )
                    )
                except Exception:
                    msg = (
                        "Could not construct receiver with passed "
                        "parameters. Check parameters for sanity."
                    )
                    raise tornado.web.HTTPError(
                        400, log_message=msg, reason=msg
                    )
        return receivers

    @tornado.gen.coroutine
    def post(self):
        if "sourcewidth" in self.request.arguments.keys():
            msg = "Parameter 'sourcewidth' is not allowed for POST requests."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Coroutine + thread as potentially pretty expensive.
        response = yield executor.submit(
            _parse_validate_and_resample_stf,
            request=self.request,
            db_info=self.application.db.info,
        )

        if isinstance(response, Exception):
            raise response

        yield executor.submit(
            self.get, custom_stf=response, nested_executor=True
        )

    @tornado.gen.coroutine
    def get(self, custom_stf=None, nested_executor=False):
        """
        :param nested_exectuor: Will not launch another executor, if true.
            Somehow tornado >= 5.0 does not like nested threads. Might be a
            good idea performance wise in any case.
        """
        # Parse the arguments. This will also perform a number of sanity
        # checks.
        args = self.parse_arguments()

        # We'll piggyback the sourcewidth on the implementation of the custom
        # STF. This is not super clean to be honest but its simple and it
        # works.
        if args.sourcewidth:
            dt = self.application.db.info.dt
            offset, data = get_gaussian_source_time_function(
                source_width=args.sourcewidth, dt=dt
            )
            custom_stf = {
                "relative_origin_time_in_sec": offset,
                "sample_spacing_in_sec": dt,
                "data": data,
            }

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

            if not isinstance(__event, dict) or sorted(
                __event.keys()
            ) != sorted(
                (
                    "m_rr",
                    "m_tt",
                    "m_pp",
                    "m_rt",
                    "m_rp",
                    "m_tp",
                    "latitude",
                    "longitude",
                    "depth_in_m",
                    "origin_time",
                )
            ):
                msg = "Event callback returned an invalid result."
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            __event["origin_time"] = obspy.UTCDateTime(__event["origin_time"])

            # In case the event is extracted, set the origin time to the
            # time of the event.
            args.origintime = __event["origin_time"]
        else:
            __event = None

        min_starttime, max_endtime = self.parse_time_settings(args)
        self.set_headers(args)

        source = self.get_source(args, __event, custom_stf=custom_stf)

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

            # Check if the connection is still open. The connection_closed
            # flag is set by the on_connection_close() method. This is
            # pretty manual right now. Maybe there is a better way? This
            # enables to server to stop serving if the connection has been
            # cancelled on the client side.
            if self.connection_closed:  # pragma: no cover
                self.flush()
                self.finish()
                return

            # Check if start- or end time are phase relative. If yes
            # calculate the new start- and/or end time.
            time_values = self.get_phase_relative_times(
                args=args,
                source=source,
                receiver=receiver,
                min_starttime=min_starttime,
                max_endtime=max_endtime,
            )
            if time_values is None:
                continue
            starttime, endtime = time_values

            # Validate the source-receiver geometry.
            self.validate_geometry(source=source, receiver=receiver)

            # Yield from the task. This enables a context switch and thus
            # async behaviour.
            if not nested_executor:
                response, mu = yield executor.submit(
                    _get_seismogram,
                    db=self.application.db,
                    source=source,
                    receiver=receiver,
                    components=list(args.components),
                    units=args.units,
                    dt=args.dt,
                    kernelwidth=args.kernelwidth,
                    starttime=starttime,
                    endtime=endtime,
                    scale=args.scale,
                    format=args.format,
                    label=args.label,
                )
            else:
                response, mu = _get_seismogram(
                    db=self.application.db,
                    source=source,
                    receiver=receiver,
                    components=list(args.components),
                    units=args.units,
                    dt=args.dt,
                    kernelwidth=args.kernelwidth,
                    starttime=starttime,
                    endtime=endtime,
                    scale=args.scale,
                    format=args.format,
                    label=args.label,
                )

            # Check connection once again.
            if self.connection_closed:  # pragma: no cover
                self.flush()
                self.finish()
                return

            # Set mu just from the first station.
            if count == 0 and not isinstance(response, Exception):
                self.set_header("Instaseis-Mu", "%f" % mu)

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
            msg = (
                "No seismograms found for the given phase relative "
                "offsets. This could either be due to the chosen phase "
                "not existing for the specific source-receiver geometry "
                "or arriving too late/with too large offsets if the "
                "database is not long enough."
            )
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Write the end of the zipfile in case necessary.
        if args.format == "saczip":
            zip_file.close()
            for data in buf:
                self.write(data)

        self.finish()

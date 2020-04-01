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
import io
import math
import numpy as np
import zipfile

import obspy
import tornado.gen
import tornado.web

from ... import FiniteSource
from ..util import IOQueue, _validtimesetting, _validate_and_write_waveforms
from ..instaseis_request import InstaseisTimeSeriesHandler
from ...source import USGSParamFileParsingException

from ...database_interfaces.base_instaseis_db import (
    KIND_MAP,
    STF_MAP,
    INV_KIND_MAP,
    _diff_and_integrate,
)


executor = concurrent.futures.ThreadPoolExecutor(12)


def _get_finite_source(
    db,
    finite_source,
    receiver,
    components,
    units,
    dt,
    kernelwidth,
    scale,
    starttime,
    endtime,
    time_of_first_sample,
    format,
    label,
):
    """
    Extract a seismogram from the passed db and write it either to a MiniSEED
    or a SACZIP file.

    :param db: An open instaseis database.
    :param finite_source: An instaseis finite source.
    :param receiver: An instaseis receiver.
    :param components: The components.
    :param units: The desired units.
    :param remove_source_shift: Remove the source time shift or not.
    :param dt: dt to resample to.
    :param kernelwidth: Width of the interpolation kernel.
    :param starttime: The desired start time of the seismogram.
    :param endtime: The desired end time of the seismogram.
    :param time_of_first_sample: The time of the first sample.
    :param format: The output format. Either "miniseed" or "saczip".
    :param label: Prefix for the filename within the SAC zip file.
    """
    try:
        st = db.get_seismograms_finite_source(
            sources=finite_source,
            receiver=receiver,
            components=components,
            # Effectively results in nothing happening so we can perform the
            # differentiation here.
            kind=INV_KIND_MAP[STF_MAP[db.info.stf]],
        )
    except Exception:
        msg = (
            "Could not extract finite source seismograms. Make sure, "
            "the parameters are valid, and the depth settings are correct."
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg), None

    for tr in st:
        tr.stats.starttime = time_of_first_sample

    finite_source.origin_time = (
        time_of_first_sample + finite_source.additional_time_shift
    )

    # Manually interpolate to get the times consistent.
    if dt:
        offset = round(finite_source.additional_time_shift % dt, 6)
        st.interpolate(
            sampling_rate=1.0 / dt,
            starttime=time_of_first_sample + offset,
            method="lanczos",
            a=kernelwidth,
            window="blackman",
        )

    # Integrate/differentiate here. No need to do it for every single
    # seismogram and stack the errors.
    n_derivative = KIND_MAP[units] - STF_MAP[db.info.stf]
    if n_derivative:
        for tr in st:
            data_summed = {}
            data_summed["A"] = tr.data
            _diff_and_integrate(
                n_derivative=n_derivative,
                data=data_summed,
                comp="A",
                dt_out=tr.stats.delta,
            )
            tr.data = data_summed["A"]

    return _validate_and_write_waveforms(
        st=st,
        scale=scale,
        starttime=starttime,
        endtime=endtime,
        source=finite_source,
        receiver=receiver,
        db=db,
        label=label,
        format=format,
    )


def _parse_and_resample_finite_source(request, db_info, max_size):
    try:
        with io.BytesIO(request.body) as buf:
            # We get 10.000 samples for each source sampled at 10 Hz. This is
            # more than enough to capture a minimal possible rise time of 1
            # second. The maximum possible time shift for any source is
            # therefore 1000 second which should be enough for any real fault.
            # Might need some more thought.
            finite_source = FiniteSource.from_usgs_param_file(
                buf, npts=10000, dt=0.1, trise_min=1.0
            )
    except USGSParamFileParsingException as e:
        msg = (
            "The body contents could not be parsed as an USGS param file "
            "due to: %s" % str(e)
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)
    # Don't forward the exception message as it might be anything and could
    # thus compromise security.
    except Exception:
        msg = (
            "Could not parse the body contents. Incorrect USGS param " "file?"
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    if max_size is not None and finite_source.npointsources > max_size:
        msg = (
            "The server only allows finite sources with at most %i points "
            "sources. The source in question has %i points."
            % (max_size, finite_source.npointsources)
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    # Check the bounds of the finite source and make sure they can be
    # calculated with the current database.
    # XXX: Also needs checks for latitude/longitude bounds if we ever
    # implement regional databases.
    min_depth = min(_i.depth_in_m for _i in finite_source.pointsources)
    max_depth = max(_i.depth_in_m for _i in finite_source.pointsources)

    db_min_depth = db_info.planet_radius - db_info.max_radius
    db_max_depth = db_info.planet_radius - db_info.min_radius

    if not (db_min_depth <= min_depth <= db_max_depth):
        msg = (
            "The shallowest point source in the given finite source is "
            "%.1f km deep. The database only has a depth range "
            "from %.1f km to %.1f km."
            % (
                min_depth / 1000.0,
                db_min_depth / 1000.0,
                db_max_depth / 1000.0,
            )
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    if not (db_min_depth <= max_depth <= db_max_depth):
        msg = (
            "The deepest point source in the given finite source is %.1f "
            "km deep. The database only has a depth range from %.1f km to "
            "%.1f km."
            % (
                max_depth / 1000.0,
                db_min_depth / 1000.0,
                db_max_depth / 1000.0,
            )
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)

    dominant_period = db_info.period

    # Here comes the magic. This is really messy but unfortunately very hard
    # to do.
    # Add two periods of samples at the beginning end the end to avoid
    # boundary effects at the ends.
    samples = int(math.ceil((2 * dominant_period / db_info.dt))) + 1
    zeros = np.zeros(samples)

    shift = samples * db_info.dt

    # We cheat a bit and set the rupture time of the first slipping patch to 0.
    # This makes aligning samples that much easier and also results in an
    # increase in maximum length of the seismograms. This has no downside
    # considering we define the origin time to the be the onset time of the
    # first slipping point source.
    first_slip = finite_source.time_shift

    for source in finite_source.pointsources:
        source.sliprate = np.concatenate([zeros, source.sliprate, zeros])
        source.time_shift += shift - first_slip

    finite_source.additional_time_shift = shift

    # A lowpass filter is needed to avoid aliasing - I guess using a
    # zerophase filter is a bit questionable as it has some potentially
    # acausal effects but it does not shift the times.
    finite_source.lp_sliprate(freq=1.0 / dominant_period, zerophase=True)

    # Last step is to resample to the sampling rate of the database for the
    # final convolution.
    finite_source.resample_sliprate(
        dt=db_info.dt, nsamp=db_info.npts + 2 * samples
    )

    # Will set the hypocentral coordinates.
    finite_source.find_hypocenter()
    return finite_source


class FiniteSourceSeismogramsHandler(InstaseisTimeSeriesHandler):
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
        # Scale parameter.
        "scale": {"type": float, "default": 1.0},
        # Receivers can be specified either directly via their coordinates.
        # In that case one can assign a network and station code.
        "receiverlatitude": {"type": float},
        "receiverlongitude": {"type": float},
        "receiverdepthinmeters": {"type": float},
        "networkcode": {"type": str, "default": "XX"},
        "stationcode": {"type": str, "default": "SYN"},
        "locationcode": {"type": str, "default": "SE"},
        # Or by querying a database.
        "network": {"type": str},
        "station": {"type": str},
        "format": {"type": str, "default": "saczip"},
    }

    default_label = "instaseis_finite_source_seismogram"
    # Done here as the time parsing is fairly complex and cannot be done
    # with normal default values.
    default_origin_time = obspy.UTCDateTime(1900, 1, 1)

    def __init__(self, *args, **kwargs):
        super(FiniteSourceSeismogramsHandler, self).__init__(*args, **kwargs)
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

        self.validate_receiver_parameters(args)

    def parse_time_settings(self, args, finite_source):
        """
        Has to be overwritten as the finite source is a bit too different.
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

        # This is now a bit of a modified clone of _get_seismogram_times()
        # of the base instaseis database object. It is modified as the
        # finite sources are a bit different.
        db = self.application.db

        time_of_first_sample = args.origintime - finite_source.time_shift

        # This is guaranteed to be exactly on a sample due to the previous
        # calculations.
        earliest_starttime = (
            time_of_first_sample + finite_source.additional_time_shift
        )
        latest_endtime = time_of_first_sample + db.info.length

        if args.dt is not None and round(args.dt / db.info.dt, 6) != 0:
            affected_area = args.kernelwidth * db.info.dt
            latest_endtime -= affected_area

        # If the endtime is not set, do it here.
        if args.endtime is None:
            args.endtime = latest_endtime

        # Do a couple of sanity checks here.
        if isinstance(args.starttime, obspy.UTCDateTime):
            # The desired seismogram start time must be before the end time of
            # the seismograms.
            if args.starttime >= latest_endtime:
                msg = "The `starttime` must be before the seismogram ends."
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)
            # Arbitrary limit: The starttime can be at max one hour before the
            # origin time.
            if args.starttime < (earliest_starttime - 3600):
                msg = (
                    "The seismogram can start at the maximum one hour "
                    "before the origin time."
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if isinstance(args.endtime, obspy.UTCDateTime):
            # The endtime must be within the seismogram window
            if not (earliest_starttime <= args.endtime <= latest_endtime):
                msg = (
                    "The end time of the seismograms lies outside the "
                    "allowed range."
                )
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        return time_of_first_sample, earliest_starttime, latest_endtime

    @tornado.gen.coroutine
    def post(self):
        # Parse the arguments. This will also perform a number of sanity
        # checks.
        args = self.parse_arguments()
        self.set_headers(args)

        # Coroutine + thread as potentially pretty expensive.
        response = yield executor.submit(
            _parse_and_resample_finite_source,
            request=self.request,
            max_size=self.application.max_size_of_finite_sources,
            db_info=self.application.db.info,
        )

        # If an exception is returned from the task, re-raise it here.
        if isinstance(response, Exception):
            raise response

        finite_source = response

        (
            time_of_first_sample,
            min_starttime,
            max_endtime,
        ) = self.parse_time_settings(args, finite_source=finite_source)

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
                source=finite_source,
                receiver=receiver,
                min_starttime=min_starttime,
                max_endtime=max_endtime,
            )
            if time_values is None:
                continue
            starttime, endtime = time_values

            # Validate the source-receiver geometry.
            self.validate_geometry(source=finite_source, receiver=receiver)

            # Yield from the task. This enables a context switch and thus
            # async behaviour.
            response, _ = yield executor.submit(
                _get_finite_source,
                db=self.application.db,
                finite_source=finite_source,
                receiver=receiver,
                components=list(args.components),
                units=args.units,
                dt=args.dt,
                kernelwidth=args.kernelwidth,
                scale=args.scale,
                starttime=starttime,
                endtime=endtime,
                time_of_first_sample=time_of_first_sample,
                format=args.format,
                label=args.label,
            )

            # Check connection once again.
            if self.connection_closed:  # pragma: no cover
                self.flush()
                self.finish()
                return

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

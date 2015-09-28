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

import obspy
import tornado.gen
import tornado.web

from ... import FiniteSource
from ..util import run_async, IOQueue, _validtimesetting, \
    _validate_and_write_waveforms
from ..instaseis_request import InstaseisTimeSeriesHandler


@run_async
def _get_finite_source(db, finite_source, receiver, components, units, dt,
                       kernelwidth, starttime, endtime, format, label,
                       origin_time, callback):
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
    :param format: The output format. Either "miniseed" or "saczip".
    :param label: Prefix for the filename within the SAC zip file.
    :param origin_time: The time of the first sample.
    :param callback: callback function of the coroutine.
    """
    try:
        st = db.get_seismograms_finite_source(
            sources=finite_source, receiver=receiver, components=components,
            kind=units, dt=dt, kernelwidth=kernelwidth)
    except Exception as e:
        print(e)
        msg = ("Could not extract seismogram. Make sure, the components "
               "are valid, and the depth settings are correct.")
        callback(tornado.web.HTTPError(400, log_message=msg, reason=msg))
        return

    for tr in st:
        tr.stats.starttime = origin_time

    finite_source.origin_time = origin_time

    _validate_and_write_waveforms(st=st, callback=callback,
                                  starttime=starttime, endtime=endtime,
                                  source=finite_source, receiver=receiver,
                                  db=db, label=label, format=format)


class FiniteSourceSeismogramsHandler(InstaseisTimeSeriesHandler):
    # Define the arguments for the seismogram endpoint.
    arguments = {
        "components": {"type": str, "default": "ZNE"},
        "units": {"type": str, "default": "displacement"},
        "dt": {"type": float},
        "kernelwidth": {"type": int, "default": 12},
        "label": {"type": str},

        # Time parameters.
        "origintime": {"type": obspy.UTCDateTime},
        "starttime": {"type": _validtimesetting,
                      "format": "Datetime String/Float/Phase+-Offset"},
        "endtime": {"type": _validtimesetting,
                    "format": "Datetime String/Float/Phase+-Offset"},

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

    default_label = "instaseis_finite_source_seismogram"
    # Done here as the time parsing is fairly complex and cannot be done
    # with normal default values.
    default_origin_time = obspy.UTCDateTime(1900, 1, 1)

    def __init__(self, *args, **kwargs):
        super(InstaseisTimeSeriesHandler, self).__init__(*args, **kwargs)

    def validate_parameters(self, args):
        """
        Function attempting to validate that the passed parameters are
        valid. Does not need to check the types as that has already been done.
        """
        self.validate_receiver_parameters(args)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        # Parse the arguments. This will also perform a number of sanity
        # checks.
        args = self.parse_arguments()
        self.set_headers(args)
        min_starttime, max_endtime = self.parse_time_settings(args)

        try:
            with io.BytesIO(self.request.body) as buf:
                finite_source = FiniteSource.from_usgs_param_file(buf)
        except:
            msg = ("Could not parse the body contents. Incorrect USGS param "
                   "file?")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        finite_source.find_hypocenter()

        # prepare the source time functions to be at the same sampling as the
        # database first use enough samples such that the lowpassed stf will
        # still be correctly represented
        dominant_period = self.application.db.info.period

        nsamp = int(dominant_period / finite_source[0].dt) * 50
        finite_source.resample_sliprate(dt=finite_source[0].dt, nsamp=nsamp)
        # lowpass to avoid aliasing
        finite_source.lp_sliprate(freq=1.0 / dominant_period)
        # finally resample to the sampling as the database
        finite_source.resample_sliprate(dt=self.application.db.info.dt,
                                        nsamp=self.application.db.info.npts)

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
            if self.connection_closed:
                self.flush()
                self.finish()
                return

            # Check if start- or end time are phase relative. If yes
            # calculate the new start- and/or end time.
            time_values = self.get_phase_relative_times(
                args=args, source=finite_source, receiver=receiver,
                min_starttime=min_starttime, max_endtime=max_endtime)
            if time_values is None:
                continue
            starttime, endtime = time_values

            # Validate the source-receiver geometry.
            # self.validate_geometry(source=finite_source, receiver=receiver)

            # Yield from the task. This enables a context switch and thus
            # async behaviour.
            response = yield tornado.gen.Task(
                _get_finite_source,
                db=self.application.db, finite_source=finite_source,
                receiver=receiver, components=list(args.components),
                units=args.units, dt=args.dt, kernelwidth=args.kernelwidth,
                starttime=starttime, endtime=endtime, format=args.format,
                label=args.label, origin_time=args.origintime)

            # Check connection once again.
            if self.connection_closed:
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

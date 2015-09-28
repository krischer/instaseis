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

from ... import __version__
from ... import FiniteSource, Receiver
from ..util import run_async, IOQueue, _validtimesetting
from ..instaseis_request import InstaseisTimeSeriesHandler
from ...helpers import geocentric_to_elliptic_latitude


@run_async
def _get_finite_source(db, finite_source, receiver, components, units, dt,
                       kernelwidth, starttime, endtime, format, label,
                       callback):
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
    :param callback: callback function of the coroutine.
    """
    if not label:
        label = ""
    else:
        label += "_"

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
        # Half the filesize but definitely sufficiently accurate.
        tr.data = np.require(tr.data, dtype=np.float32)

    # XXX: Reenable once figured out!
    # Sanity checks. Raise internal server errors in case something fails.
    # This should not happen and should have been caught before.
    # if endtime > st[0].stats.endtime:
    #     msg = ("Endtime larger then the extracted endtime: endtime=%s, "
    #            "largest db endtime=%s" % (endtime, st[0].stats.endtime))
    #     callback(tornado.web.HTTPError(500, log_message=msg, reason=msg))
    #     return
    # if starttime < st[0].stats.starttime - 3600.0:
    #     msg = ("Starttime more than one hour before the starttime of the "
    #            "seismograms.")
    #     callback(tornado.web.HTTPError(500, log_message=msg, reason=msg))
    #     return
    #
    # # Trim, potentially pad with zeroes.
    # st.trim(starttime, endtime, pad=True, fill_value=0.0, nearest_sample=False)

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
            tr.stats.sac.evla = geocentric_to_elliptic_latitude(
                finite_source.hypocenter_latitude)
            tr.stats.sac.evlo = finite_source.hypocenter_longitude
            tr.stats.sac.evdp = finite_source.hypocenter_depth_in_m
            tr.stats.sac.mag = finite_source.moment_magnitude
            # Thats what SPECFEM uses for a moment magnitude....
            tr.stats.sac.imagtyp = 55
            # The event origin time relative to the reference which I'll
            # just assume to be the starttime here?
            # XXX: Reenable once figured out!
            # tr.stats.sac.o = source.origin_time - starttime
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
        # The networkcode and stationcode parameters have a maximum number
        # of letters.
        if args.stationcode and len(args.stationcode) > 5:
            msg = "'stationcode' must have 5 or fewer letters."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if args.networkcode and len(args.networkcode) > 2:
            msg = "'networkcode' must have 2 or fewer letters."
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Figure out how the station coordinates are specified.
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

    def get_receivers(self, args):
        # Already checked before - just make sure the settings are valid.
        assert (args.receiverlatitude is not None and
                args.receiverlongitude is not None) or \
           (args.network and args.station)

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
        return receivers

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        # Parse the arguments. This will also perform a number of sanity
        # checks.
        args = self.parse_arguments()

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


        min_starttime, max_endtime = self.parse_time_settings(args)
        self.set_headers(args)

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
                label=args.label)

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

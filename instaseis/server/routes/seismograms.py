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
from ..util import run_async
from ..instaseis_request import InstaseisRequestHandler


@run_async
def _get_seismogram(db, source, receiver, components, unit,
                    remove_source_shift, dt, a_lanczos, format, callback):
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
    :param format:
    :param callback: callback function of the coroutine.
    """
    try:
        st = db.get_seismograms(
            source=source, receiver=receiver, components=components,
            kind=unit, remove_source_shift=remove_source_shift,
            reconvolve_stf=False, return_obspy_stream=True, dt=dt,
            a_lanczos=a_lanczos)
    except Exception:
        msg = ("Could not extract seismogram. Make sure, the components "
               "are valid, and the depth settings are correct.")
        callback(tornado.web.HTTPError(400, log_message=msg, reason=msg))
        return

    # Half the filesize but definitely sufficiently accurate.
    for tr in st:
        tr.data = np.require(tr.data, dtype=np.float32)

    if format == "mseed":
        with io.BytesIO() as fh:
            st.write(fh, format="mseed")
            fh.seek(0, 0)
            binary_data = fh.read()
    # Write a number of SAC files into an archive.
    elif format == "saczip":
        with io.BytesIO() as fh:
            with zipfile.ZipFile(fh, mode="w") as zh:
                for tr in st:
                    with io.BytesIO() as temp:
                        tr.write(temp, format="sac")
                        temp.seek(0, 0)
                        filename = "%s.sac" % tr.id
                        zh.writestr(filename, temp.read())
            fh.seek(0, 0)
            binary_data = fh.read()
    else:
        # Checked above and cannot really happen.
        raise NotImplementedError

    callback(binary_data)


class SeismogramsHandler(InstaseisRequestHandler):
    # Define the arguments for the seismogram endpoint.
    seismogram_arguments = {
        "components": {"type": str, "default": "ZNE"},
        "unit": {"type": str, "default": "displacement"},
        "removesourceshift": {"type": bool, "default": True},
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
        # More optional source parameters.
        "origintime": {"type": obspy.UTCDateTime,
                       "default": obspy.UTCDateTime(0)},
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

    def parse_arguments(self):
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
                    t = properties["type"]
                    if t is bool and not isinstance(value, bool):
                        if value.lower() in ["1", "true", "t", "y"]:
                            value = True
                        elif value.lower() in ["0", "false", "f", "n"]:
                            value = False
                        else:
                            raise ValueError
                    else:
                        value = properties["type"](value)
                except:
                    msg = "Parameter '%s' could not be converted to '%s'." % (
                        name, str(properties["type"].__name__))
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            setattr(args, name, value)
        return args

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
                                    m_rp=args.mrp, m_tp=args.mtp,
                                    origin_time=args.origintime)
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
                        M0=args.M0, origin_time=args.origintime)
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
                                         f_p=args.fp,
                                         origin_time=args.origintime)
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

        # For each, get the synthetics, and stream it to the user.
        for receiver in receivers:
            # Yield from the task. This enables a context switch and thus
            # async behaviour.
            response = yield tornado.gen.Task(
                _get_seismogram,
                db=self.application.db, source=source, receiver=receiver,
                remove_source_shift=args.removesourceshift,
                components=components,  unit=args.unit, dt=args.dt,
                a_lanczos=args.alanczos, format=args.format)

            if isinstance(response, Exception):
                raise response

            self.write(response)
            self.flush()

        FILE_ENDINGS_MAP = {
            "mseed": "mseed",
            "saczip": "zip"}

        filename = "instaseis_seismogram_%s.%s" % (
            str(obspy.UTCDateTime()).replace(":", "_"),
            FILE_ENDINGS_MAP[args.format])

        if format == "mseed":
            content_type = "application/octet-stream"
        elif format == "saczip":
            content_type = "application/zip"
        self.set_header("Content-Type", content_type)

        self.set_header("Content-Disposition",
                        "attachment; filename=%s" % filename)
        self.finish()

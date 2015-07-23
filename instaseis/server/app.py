#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Server offering a REST API for Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import copy
import functools
import io
import logging
import threading
import zipfile

import numpy as np
import obspy
import tornado.gen
import tornado.ioloop
import tornado.web

from ..import __version__
from ..instaseis_db import InstaseisDB
from .. import Source, ForceSource, Receiver


def run_async(func):
    @functools.wraps(func)
    def async_func(*args, **kwargs):
        func_hl = threading.Thread(target=func, args=args, kwargs=kwargs)
        func_hl.start()
        return func_hl
    return async_func


@run_async
def _get_seismogram(db, source, receiver, components, unit,
                    remove_source_shift, dt, a_lanczos, format, callback):
    try:
        st = db.get_seismograms(
            source=source, receiver=receiver, components=components,
            kind=unit, remove_source_shift=remove_source_shift,
            reconvolve_stf=False, return_obspy_stream=True, dt=dt,
            a_lanczos=a_lanczos)
    except Exception:
        msg = ("Could not extract seismogram. Make sure, the components "
               "are valid, and the depth settings are correct.")
        raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

    # Half the filesize but definitely sufficiently accurate.
    for tr in st:
        tr.data = np.require(tr.data, dtype=np.float32)

    if format == "mseed":
        with io.BytesIO() as fh:
            st.write(fh, format="mseed")
            fh.seek(0, 0)
            binary_data = fh.read()
        content_type = "application/octet-stream"
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
        content_type = "application/zip"
    else:
        # Checked above and cannot really happen.
        raise NotImplementedError

    callback(binary_data, content_type)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        response = {
            "type": "Instaseis Remote Server",
            "version": __version__
        }
        self.write(response)
        self.set_header("Access-Control-Allow-Origin", "*")


class InfoHandler(tornado.web.RequestHandler):
    def get(self):
        info = copy.deepcopy(application.db.info)
        # No need to write a custom encoder...
        info["datetime"] = str(info["datetime"])
        info["slip"] = list([float(_i) for _i in info["slip"]])
        info["sliprate"] = list([float(_i) for _i in info["sliprate"]])
        # Clear the directory to avoid leaking any more system information then
        # necessary.
        info["directory"] = ""
        self.write(dict(info))
        self.set_header("Access-Control-Allow-Origin", "*")


class SeismogramsHandler(tornado.web.RequestHandler):
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
                not application.station_coordinates_callback:
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

            coordinates = application.station_coordinates_callback(
                networks=networks, stations=stations)

            if not coordinates:
                msg = "No coordinates found satisfying the query."
                raise tornado.web.HTTPError(
                    404, log_message=msg, reason=msg)
                application.station_coordinates_callback(networks=args.network,
                                                         stations=args.station)

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

            response, _ = yield tornado.gen.Task(
                _get_seismogram,
                db=application.db, source=source, receiver=receiver,
                remove_source_shift=args.removesourceshift,
                components=components,  unit=args.unit, dt=args.dt,
                a_lanczos=args.alanczos, format=args.format)

            binary_data, content_type = response

            self.write(binary_data)
            self.flush()

        FILE_ENDINGS_MAP = {
            "mseed": "mseed",
            "saczip": "zip"}

        filename = "instaseis_seismogram_%s.%s" % (
            str(obspy.UTCDateTime()).replace(":", "_"),
            FILE_ENDINGS_MAP[args.format])

        self.set_header("Content-Type", content_type)
        self.set_header("Content-Disposition",
                        "attachment; filename=%s" % filename)
        self.set_header("Access-Control-Allow-Origin", "*")
        self.finish()


class RawSeismogramsHandler(tornado.web.RequestHandler):
    # Define the arguments for the seismogram endpoint.
    seismogram_arguments = {
        "components": {"type": str, "default": "ZNE"},
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
        # Receiver parameters.
        "receiverlatitude": {"type": float, "required": True},
        "receiverlongitude": {"type": float, "required": True},
        "receiverdepthinm": {"type": float, "default": 0.0},
        "networkcode": {"type": str},
        "stationcode": {"type": str}
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
                    value = properties["type"](value)
                except:
                    msg = "Parameter '%s' could not be converted to '%s'." % (
                        name, str(properties["type"]))
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            setattr(args, name, value)
        return args

    def get(self):
        args = self.parse_arguments()

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

        # Construct the receiver object.
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

        # Get the most barebones seismograms possible.
        try:
            application.db._get_seismograms_sanity_checks(
                source=source, receiver=receiver, components=components,
                kind="displacement")
            data = application.db._get_seismograms(
                source=source, receiver=receiver, components=components)
        except Exception:
            msg = ("Could not extract seismogram. Make sure, the components "
                   "are valid, and the depth settings are correct.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        try:
            st = application.db._convert_to_stream(
                source=source, receiver=receiver, components=components,
                data=data, dt_out=application.db.info.dt)
        except Exception:
            msg = ("Could not convert seismogram to a Stream object.")
            raise tornado.web.HTTPError(500, log_message=msg, reason=msg)

        # Half the filesize but definitely sufficiently accurate.
        for tr in st:
            tr.data = np.require(tr.data, dtype=np.float32)

        with io.BytesIO() as fh:
            st.write(fh, format="mseed")
            fh.seek(0, 0)
            binary_data = fh.read()

        filename = "instaseis_seismogram_%s.mseed" % \
                   str(obspy.UTCDateTime()).replace(":", "_")

        self.write(binary_data)

        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Content-Disposition",
                        "attachment; filename=%s" % filename)
        # Passing mu in the HTTP header...not sure how well this plays with
        # proxies...
        self.set_header("Instaseis-Mu", "%f" % st[0].stats.instaseis.mu)
        self.set_header("Access-Control-Allow-Origin", "*")


class CoordinatesHandler(tornado.web.RequestHandler):
    def get(self):
        if application.station_coordinates_callback is None:
            msg = "Server does not support station coordinates."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        networks = self.get_argument("network")
        stations = self.get_argument("station")

        if not networks or not stations:
            msg = "Parameters 'network' and 'station' must be given."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        networks = networks.split(",")
        stations = stations.split(",")

        coordinates = application.station_coordinates_callback(
            networks=networks, stations=stations)

        if not coordinates:
            msg = "No coordinates found satisfying the query."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        self.write({"count": len(coordinates), "stations": coordinates})
        self.set_header("Access-Control-Allow-Origin", "*")


application = tornado.web.Application([
    (r"/seismograms", SeismogramsHandler),
    (r"/seismograms_raw", RawSeismogramsHandler),
    (r"/info", InfoHandler),
    (r"/", IndexHandler),
    (r"/coordinates", CoordinatesHandler),
])


def launch_io_loop(db_path, port, buffer_size_in_mb, quiet, log_level,
                   station_coordinates_callback=None):
    application.db = InstaseisDB(db_path=db_path,
                                 buffer_size_in_mb=buffer_size_in_mb)
    application.station_coordinates_callback = station_coordinates_callback

    if not quiet:
        # Get all tornado loggers.
        access_log = logging.getLogger("tornado.access")
        app_log = logging.getLogger("tornado.application")
        gen_log = logging.getLogger("tornado.general")
        loggers = (access_log, app_log, gen_log)

        # Console log handler.
        ch = logging.StreamHandler()
        # Add formatter
        FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(FORMAT)
        ch.setFormatter(formatter)

        log_level = getattr(logging, log_level)

        for logger in loggers:
            logger.addHandler(ch)
            logger.setLevel(log_level)

        # Log the database information.
        app_log.info("Successfully opened DB")
        app_log.info(str(application.db))

    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

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
import io
import logging
import numpy as np
import obspy
import tornado.ioloop
import tornado.web

from ..import __version__
from ..instaseis_db import InstaseisDB
from .. import Source, ForceSource, Receiver


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        response = {
            "type": "Instaseis Remote Server",
            "version": __version__
        }
        self.write(response)


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


class SeismogramsHandler(tornado.web.RequestHandler):
    # Define the arguments for the seismogram endpoint.
    seismogram_arguments = {
        "components": {"type": str, "default": "ZNE"},
        "unit": {"type": str, "default": "displacement"},
        "remove_source_shift": {"type": bool, "default": True},
        "dt": {"type": float},
        "a_lanczos": {"type": int, "default": 5},
        # Source parameters.
        "source_latitude": {"type": float, "required": True},
        "source_longitude": {"type": float, "required": True},
        "source_depth_in_m": {"type": float, "default": 0.0},
        # Source can either be given as the moment tensor components in Nm.
        "m_rr": {"type": float},
        "m_tt": {"type": float},
        "m_pp": {"type": float},
        "m_rt": {"type": float},
        "m_rp": {"type": float},
        "m_tp": {"type": float},
        # Or as strike, dip, rake and M0.
        "strike": {"type": float},
        "dip": {"type": float},
        "rake": {"type": float},
        "M0": {"type": float},
        # Or as a force source.
        "f_r": {"type": float},
        "f_t": {"type": float},
        "f_p": {"type": float},
        # More optional source parameters.
        "origin_time": {"type": obspy.UTCDateTime,
                        "default": obspy.UTCDateTime(0)},
        # Receiver parameters.
        "receiver_latitude": {"type": float, "required": True},
        "receiver_longitude": {"type": float, "required": True},
        "receiver_depth_in_m": {"type": float, "default": 0.0},
        "network_code": {"type": str},
        "station_code": {"type": str}
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

    def get(self):
        args = self.parse_arguments()

        # Make sure the unit arguments is valid.
        args.unit = args.unit.lower()
        if args.unit not in ["displacement", "velocity", "acceleration"]:
            msg = ("Unit must be one of 'displacement', 'velocity', "
                   "or 'acceleration'")
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
        if not (2 <= args.a_lanczos <= 20):
            msg = ("`a_lanczos` must not be smaller than 2 or larger than 20.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Figure out the type of source and construct the source object.
        src_params = {
            "moment_tensor": set(["m_rr", "m_tt", "m_pp", "m_rt", "m_rp",
                                  "m_tp"]),
            "strike_dip_rake": set(["strike", "dip", "rake", "M0"]),
            "force_source": set(["f_r", "f_t", "f_p"])
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
                    source = Source(latitude=args.source_latitude,
                                    longitude=args.source_longitude,
                                    depth_in_m=args.source_depth_in_m,
                                    m_rr=args.m_rr, m_tt=args.m_tt,
                                    m_pp=args.m_pp, m_rt=args.m_rt,
                                    m_rp=args.m_rp, m_tp=args.m_tp,
                                    origin_time=args.origin_time)
                except:
                    msg = ("Could not construct moment tensor source with "
                           "passed parameters. Check parameters for sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
                break
            elif src_type == "strike_dip_rake":
                try:
                    source = Source.from_strike_dip_rake(
                        latitude=args.source_latitude,
                        longitude=args.source_longitude,
                        depth_in_m=args.source_depth_in_m,
                        strike=args.strike, dip=args.dip, rake=args.rake,
                        M0=args.M0, origin_time=args.origin_time)
                except:
                    msg = ("Could not construct the source from the passed "
                           "strike/dip/rake parameters. Check parameter for "
                           "sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
                break
            elif src_type == "force_source":
                try:
                    source = ForceSource(latitude=args.source_latitude,
                                         longitude=args.source_longitude,
                                         depth_in_m=args.source_depth_in_m,
                                         f_r=args.f_r, f_t=args.f_t,
                                         f_p=args.f_p,
                                         origin_time=args.origin_time)
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
            receiver = Receiver(latitude=args.receiver_latitude,
                                longitude=args.receiver_longitude,
                                network=args.network_code,
                                station=args.station_code,
                                depth_in_m=args.receiver_depth_in_m)
        except:
            msg = ("Could not construct receiver with passed parameters. "
                   "Check parameters for sanity.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        try:
            st = application.db.get_seismograms(
                source=source, receiver=receiver, components=components,
                kind=args.unit, remove_source_shift=args.remove_source_shift,
                reconvolve_stf=False, return_obspy_stream=True, dt=args.dt,
                a_lanczos=args.a_lanczos)
        except Exception:
            msg = ("Could not extract seismogram. Make sure, the components "
                   "are valid, and the depth settings are correct.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

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


class RawSeismogramsHandler(tornado.web.RequestHandler):
    # Define the arguments for the seismogram endpoint.
    seismogram_arguments = {
        "components": {"type": str, "default": "ZNE"},
        # Source parameters.
        "source_latitude": {"type": float, "required": True},
        "source_longitude": {"type": float, "required": True},
        "source_depth_in_m": {"type": float, "default": 0.0},
        # Source can either be given as the moment tensor components in Nm.
        "m_rr": {"type": float},
        "m_tt": {"type": float},
        "m_pp": {"type": float},
        "m_rt": {"type": float},
        "m_rp": {"type": float},
        "m_tp": {"type": float},
        # Or as strike, dip, rake and M0.
        "strike": {"type": float},
        "dip": {"type": float},
        "rake": {"type": float},
        "M0": {"type": float},
        # Or as a force source.
        "f_r": {"type": float},
        "f_t": {"type": float},
        "f_p": {"type": float},
        # More optional source parameters.
        "origin_time": {"type": obspy.UTCDateTime,
                        "default": obspy.UTCDateTime(0)},
        # Receiver parameters.
        "receiver_latitude": {"type": float, "required": True},
        "receiver_longitude": {"type": float, "required": True},
        "receiver_depth_in_m": {"type": float, "default": 0.0},
        "network_code": {"type": str},
        "station_code": {"type": str}
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
            "moment_tensor": set(["m_rr", "m_tt", "m_pp", "m_rt", "m_rp",
                                  "m_tp"]),
            "strike_dip_rake": set(["strike", "dip", "rake", "M0"]),
            "force_source": set(["f_r", "f_t", "f_p"])
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
                    source = Source(latitude=args.source_latitude,
                                    longitude=args.source_longitude,
                                    depth_in_m=args.source_depth_in_m,
                                    m_rr=args.m_rr, m_tt=args.m_tt,
                                    m_pp=args.m_pp, m_rt=args.m_rt,
                                    m_rp=args.m_rp, m_tp=args.m_tp,
                                    origin_time=args.origin_time)
                except:
                    msg = ("Could not construct moment tensor source with "
                           "passed parameters. Check parameters for sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
                break
            elif src_type == "strike_dip_rake":
                try:
                    source = Source.from_strike_dip_rake(
                        latitude=args.source_latitude,
                        longitude=args.source_longitude,
                        depth_in_m=args.source_depth_in_m,
                        strike=args.strike, dip=args.dip, rake=args.rake,
                        M0=args.M0, origin_time=args.origin_time)
                except:
                    msg = ("Could not construct the source from the passed "
                           "strike/dip/rake parameters. Check parameter for "
                           "sanity.")
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
                break
            elif src_type == "force_source":
                try:
                    source = ForceSource(latitude=args.source_latitude,
                                         longitude=args.source_longitude,
                                         depth_in_m=args.source_depth_in_m,
                                         f_r=args.f_r, f_t=args.f_t,
                                         f_p=args.f_p,
                                         origin_time=args.origin_time)
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
            receiver = Receiver(latitude=args.receiver_latitude,
                                longitude=args.receiver_longitude,
                                network=args.network_code,
                                station=args.station_code,
                                depth_in_m=args.receiver_depth_in_m)
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


application = tornado.web.Application([
    (r"/seismograms", SeismogramsHandler),
    (r"/seismograms_raw", RawSeismogramsHandler),
    (r"/info", InfoHandler),
    (r"/", IndexHandler)
])


def launch_io_loop(db_path, port, buffer_size_in_mb, quiet, log_level):
    application.db = InstaseisDB(db_path=db_path,
                                 buffer_size_in_mb=buffer_size_in_mb)

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

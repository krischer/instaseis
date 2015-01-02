#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Server offering a REST API for Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import copy
import io
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
        # Source parameters.
        "source_latitude": {"type": float, "required": True},
        "source_longitude": {"type": float, "required": True},
        "source_depth_in_m": {"type": float},
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
        "receiver_depth_in_m": {"type": float},
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
                    raise tornado.web.HTTPError(
                        400,
                        reason="Required parameter '%s' not given." % name)
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
                    raise tornado.web.HTTPError(
                        400, reason="Parameter '%s' could not be converted "
                        "to '%s'." % (name, str(properties["type"])))
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
                    raise tornado.web.HTTPError(
                        400, reason="Could not construct moment tensor source "
                        "with passed parameters. Check parameters for sanity.")
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
                    raise tornado.web.HTTPError(
                        400, reason="Could not construct the source from the "
                        "passed strike/dip/rake parameters. Check parameter "
                        "for sanity.")
                break
            elif src_type == "force_source":
                try:
                    source = ForceSource(latitude=args.source_latitude,
                                         longitude=args.source_longitude,
                                         depth_in_m=args.source_depth_in_m,
                                         f_r=args.f_r, f_t=args.f_t,
                                         f_p=args.f_p)
                except:
                    raise tornado.web.HTTPError(
                        400, reason="Could not construct force source with "
                        "passed parameters. Check parameters for sanity.")
                break
            else:
                # Cannot really happen.
                raise NotImplementedError
        else:
            raise tornado.web.HTTPError(
                400, reason="No/insufficient source parameters specified")

        # Construct the receiver object.
        try:
            receiver = Receiver(latitude=args.receiver_latitude,
                                longitude=args.receiver_longitude,
                                network=args.network_code,
                                station=args.station_code,
                                depth_in_m=args.receiver_depth_in_m)
        except:
            raise tornado.web.HTTPError(
                400, reason="Could not construct receiver with passed "
                "parameters. Check parameters for sanity.")

        # Get the most barebones seismograms possible.
        try:
            st = application.db.get_seismograms(
                source=source, receiver=receiver, components=components,
                kind="displacement", remove_source_shift=False,
                reconvolve_stf=False, return_obspy_stream=True, dt=None)
        except Exception:
            raise tornado.web.HTTPError(
                400, reason="Could not extract seismogram. Make sure, "
                "the components are valid, and the depth settings are "
                "correct.")

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
        self.set_header("Instaseis-Mu", str(st[0].stats.instaseis.mu))


application = tornado.web.Application([
    (r"/seismograms_raw", SeismogramsHandler),
    (r"/info", InfoHandler),
    (r"/", IndexHandler)
])


def launch_io_loop(db_path, port, buffer_size_in_mb):
    application.db = InstaseisDB(db_path=db_path,
                                 buffer_size_in_mb=buffer_size_in_mb)
    print(application.db)
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

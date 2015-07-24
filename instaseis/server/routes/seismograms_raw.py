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

import numpy as np
import obspy
import tornado.web

from ... import Source, ForceSource, Receiver


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
            self.application.db._get_seismograms_sanity_checks(
                source=source, receiver=receiver, components=components,
                kind="displacement")
            data = self.application.db._get_seismograms(
                source=source, receiver=receiver, components=components)
        except Exception:
            msg = ("Could not extract seismogram. Make sure, the components "
                   "are valid, and the depth settings are correct.")
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        try:
            st = self.application.db._convert_to_stream(
                source=source, receiver=receiver, components=components,
                data=data, dt_out=self.application.db.info.dt)
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

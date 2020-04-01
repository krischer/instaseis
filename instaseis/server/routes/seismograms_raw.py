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

import numpy as np
import obspy
import tornado.web

from ... import Source, ForceSource, Receiver
from ..instaseis_request import InstaseisTimeSeriesHandler

executor = concurrent.futures.ThreadPoolExecutor(12)


def _get_seismogram(db, source, receiver, components):
    """
    Extract a seismogram from the passed db and write it either to a MiniSEED
    or a SACZIP file.

    :param db: An open instaseis database.
    :param source: An instaseis source.
    :param receiver: An instaseis receiver.
    :param components: The components.
    :param callback: callback function of the coroutine.
    """
    # Get the most barebones seismograms possible.
    try:
        # We extract seismograms at the database sampling rate - thus
        # setting it to None is alright here.
        db._get_seismograms_sanity_checks(
            source=source,
            receiver=receiver,
            components=components,
            kind="displacement",
            dt=None,
        )
        data = db._get_seismograms(
            source=source, receiver=receiver, components=components
        )
    except Exception:
        msg = (
            "Could not extract seismogram. Make sure, the components "
            "are valid, and the depth settings are correct."
        )
        return tornado.web.HTTPError(400, log_message=msg, reason=msg)
        return

    try:
        st = db._convert_to_stream(
            receiver=receiver,
            components=components,
            data=data,
            dt_out=db.info.dt,
            starttime=source.origin_time,
        )
    except Exception:
        msg = "Could not convert seismogram to a Stream object."
        return tornado.web.HTTPError(500, log_message=msg, reason=msg)

    # Half the filesize but definitely sufficiently accurate.
    for tr in st:
        tr.data = np.require(tr.data, dtype=np.float32)

    with io.BytesIO() as fh:
        st.write(fh, format="mseed")
        fh.seek(0, 0)
        binary_data = fh.read()
    return binary_data, st[0].stats.instaseis.mu


class RawSeismogramsHandler(InstaseisTimeSeriesHandler):
    # Define the arguments for the seismogram endpoint.
    arguments = {
        # Default arguments are either 'ZNE', 'Z', or 'NE', depending on
        # what the database supports. Default argument will be set later when
        # the database is known.
        "components": {"type": str},
        # Source parameters.
        "sourcelatitude": {"type": float, "required": True},
        "sourcelongitude": {"type": float, "required": True},
        "sourcedepthinmeters": {"type": float, "default": 0.0},
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
        "origintime": {
            "type": obspy.UTCDateTime,
            "default": obspy.UTCDateTime(0),
        },
        # Receiver parameters.
        "receiverlatitude": {"type": float, "required": True},
        "receiverlongitude": {"type": float, "required": True},
        "receiverdepthinmeters": {"type": float, "default": 0.0},
        "networkcode": {"type": str},
        "stationcode": {"type": str},
        "locationcode": {"type": str},
    }
    default_label = "instaseis_seismogram"

    def validate_parameters(self, args):
        pass

    def __init__(self, *args, **kwargs):
        super(RawSeismogramsHandler, self).__init__(*args, **kwargs)
        # Set the correct default arguments.
        self.arguments["components"]["default"] = "".join(
            self.application.db.default_components
        )

    @tornado.gen.coroutine
    def get(self):
        args = self.parse_arguments()

        # Figure out the type of source and construct the source object.
        src_params = {
            "moment_tensor": set(["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]),
            "strike_dip_rake": set(["strike", "dip", "rake", "M0"]),
            "force_source": set(["fr", "ft", "fp"]),
        }

        components = list(args.components)
        for src_type, params in src_params.items():
            src_params = [getattr(args, _i) for _i in params]
            if None in src_params:
                continue
            elif src_type == "moment_tensor":
                try:
                    source = Source(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        m_rr=args.mrr,
                        m_tt=args.mtt,
                        m_pp=args.mpp,
                        m_rt=args.mrt,
                        m_rp=args.mrp,
                        m_tp=args.mtp,
                        origin_time=args.origintime,
                    )
                except Exception:
                    msg = (
                        "Could not construct moment tensor source with "
                        "passed parameters. Check parameters for sanity."
                    )
                    raise tornado.web.HTTPError(
                        400, log_message=msg, reason=msg
                    )
                break
            elif src_type == "strike_dip_rake":
                try:
                    source = Source.from_strike_dip_rake(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        strike=args.strike,
                        dip=args.dip,
                        rake=args.rake,
                        M0=args.M0,
                        origin_time=args.origintime,
                    )
                except Exception:
                    msg = (
                        "Could not construct the source from the passed "
                        "strike/dip/rake parameters. Check parameter for "
                        "sanity."
                    )
                    raise tornado.web.HTTPError(
                        400, log_message=msg, reason=msg
                    )
                break
            elif src_type == "force_source":
                try:
                    source = ForceSource(
                        latitude=args.sourcelatitude,
                        longitude=args.sourcelongitude,
                        depth_in_m=args.sourcedepthinmeters,
                        f_r=args.fr,
                        f_t=args.ft,
                        f_p=args.fp,
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
                break
            else:
                # Cannot really happen.
                raise NotImplementedError
        else:
            msg = "No/insufficient source parameters specified"
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Construct the receiver object.
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

        response = yield executor.submit(
            _get_seismogram,
            db=self.application.db,
            source=source,
            receiver=receiver,
            components=components,
        )

        # If an exception is returned from the task, re-raise it here.
        if isinstance(response, Exception):
            raise response

        self.set_headers(args)
        # Passing mu in the HTTP header...not sure how well this plays with
        # proxies...
        self.set_header("Instaseis-Mu", "%f" % response[1])

        self.write(response[0])
        self.finish()

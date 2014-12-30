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
import flask
from flask import Flask, make_response
from flask.ext.restful import reqparse
import io
import numpy as np
import obspy

from ..instaseis_db import InstaseisDB
from .. import Source, ForceSource, Receiver


class InvalidSourceError(Exception):
    pass


# Dictionary responsible for pretty error messages with various HTTP codes.
errors = {
    "InvalidSourceError": {
        "message": "The source definition is invalid.",
        "status": 404,
        }
}


app = Flask(__name__)


@app.route("/")
def index():
    desc = {
        "type": "Instaseis Remote Server"
    }
    return flask.jsonify(**desc)


@app.route("/info")
def info():
    info = copy.deepcopy(app.db.info)
    # No need to write a custom encoder...
    info["datetime"] = str(info["datetime"])
    info["slip"] = list([float(_i) for _i in info["slip"]])
    info["sliprate"] = list([float(_i) for _i in info["sliprate"]])
    # Clear the directory to avoid leaking any more system information then
    # necessary.
    info["directory"] = ""
    return flask.jsonify(**info)


# Define the seismogram parser.
seismogram_parser = reqparse.RequestParser()
# Source parameters.
seismogram_parser.add_argument("source_latitude", type=float, required=True,
                               help="source_latitude is required")
seismogram_parser.add_argument("source_longitude", type=float, required=True,
                               help="source_longitude is required")
seismogram_parser.add_argument("source_depth_in_m", type=float)
# Source can either be given as the moment tensor components in Nm.
seismogram_parser.add_argument("m_rr", type=float)
seismogram_parser.add_argument("m_tt", type=float)
seismogram_parser.add_argument("m_pp", type=float)
seismogram_parser.add_argument("m_rt", type=float)
seismogram_parser.add_argument("m_rp", type=float)
seismogram_parser.add_argument("m_tp", type=float)
# Or as strike, dip, rake and M0.
seismogram_parser.add_argument("strike", type=float)
seismogram_parser.add_argument("dip", type=float)
seismogram_parser.add_argument("rake", type=float)
seismogram_parser.add_argument("M0", type=float)
# Or as a force source.
seismogram_parser.add_argument("f_r", type=float)
seismogram_parser.add_argument("f_t", type=float)
seismogram_parser.add_argument("f_p", type=float)
# More optional source parameters.
seismogram_parser.add_argument("source_sliprate", type=float)
seismogram_parser.add_argument("stf_dt", type=float)
seismogram_parser.add_argument("origin_time", type=obspy.UTCDateTime,
                               default=obspy.UTCDateTime(0))
# Receiver parameters.
seismogram_parser.add_argument("receiver_latitude", type=float, required=True,
                               help="receiver_latitude is required")
seismogram_parser.add_argument("receiver_longitude", type=float, required=True,
                               help="receiver_longitude is required")
seismogram_parser.add_argument("receiver_depth_in_m", type=float)
seismogram_parser.add_argument("network_code", type=str)
seismogram_parser.add_argument("station_code", type=str)


@app.route("/seismograms", methods=["GET", "POST"])
def get_seismograms():
    args = seismogram_parser.parse_args()
    print(args)

    # Figure out the type of source and construct the source object.
    src_params = {
        "moment_tensor": set(["m_rr", "m_tt", "m_pp", "m_rt", "m_rp",
                              "m_tp"]),
        "strike_dip_rake": set(["strike", "dip", "rake", "M0"]),
        "force_source": set(["f_r", "f_t", "f_p"])
    }
    for src_type, params in src_params.items():
        src_params = [getattr(args, _i) for _i in params]
        if None in src_params:
            continue
        elif src_type == "moment_tensor":
            source = Source(latitude=args.source_latitude,
                            longitude=args.source_longitude,
                            depth_in_m=args.source_depth_in_m,
                            m_rr=args.m_rr, m_tt=args.m_tt, m_pp=args.m_pp,
                            m_rt=args.m_rt, m_rp=args.m_rp, m_tp=args.m_tp,
                            sliprate=args.source_sliprate,
                            dt=args.stf_dt, origin_time=args.origin_time)
            break
        elif src_type == "strike_dip_rake":
            source = Source.from_strike_dip_rake(
                latitude=args.source_latitude,
                longitude=args.source_longitude,
                depth_in_m=args.source_depth_in_m,
                strike=args.strike, dip=args.dip, rake=args.rake,
                M0=args.M0, sliprate=args.source_sliprate,
                dt=args.stf_dt, origin_time=args.origin_time)
            break
        elif src_type == "force_source":
            source = ForceSource(latitude=args.source_latitude,
                                 longitude=args.source_longitude,
                                 depth_in_m=args.source_depth_in_m,
                                 f_r=args.f_r, f_t=args.f_t, f_p=args.f_p)
            break
        else:
            raise InvalidSourceError
    else:
        raise InvalidSourceError

    # Construct the receiver object.
    receiver = Receiver(latitude=args.receiver_latitude,
                        longitude=args.receiver_longitude,
                        network=args.network_code,
                        station=args.station_code,
                        depth_in_m=args.receiver_depth_in_m)

    st = app.db.get_seismograms(source=source, receiver=receiver)
    # Half the filesize but definitely sufficiently accurate.
    for tr in st:
        tr.data = np.require(tr.data, dtype=np.float32)

    with io.BytesIO() as fh:
        st.write(fh, format="mseed")
        fh.seek(0, 0)
        binary_data = fh.read()

    filename = "instaseis_seismogram_%s.mseed" % \
        str(obspy.UTCDateTime()).replace(":", "_")

    response = make_response(binary_data)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Content-Disposition"] = \
        "attachment; filename=%s" % filename
    # Passing mu in the HTTP header...not sure how well this plays with
    # proxies...
    response.headers["Instaseis-Mu"] = st[0].stats.instaseis.mu

    return response


def serve(db_path, port, buffer_size_in_mb):
    app.db = InstaseisDB(db_path=db_path, buffer_size_in_mb=buffer_size_in_mb)
    print(app.db)
    app.run(host="0.0.0.0", port=port, debug=True)

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
import flask
from flask import Flask, request
from flask.ext import restful
from flask.ext.restful import reqparse

import obspy

from ..instaseisdb import InstaseisDB

app = Flask(__name__)
api = restful.Api(app)


@app.route("/")
def index():
    info = app.db.info
    # No need to write a custom encoder...
    info["datetime"] = str(info["datetime"])
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
seismogram_parser.add_argument("origin_time", type=str)
# Receiver parameters.
seismogram_parser.add_argument("receiver_latitude", type=float, required=True,
                               help="receiver_latitude is required")
seismogram_parser.add_argument("receiver_longitude", type=float, required=True,
                               help="receiver_longitude is required")
seismogram_parser.add_argument("receiver_depth_in_m", type=float)
seismogram_parser.add_argument("network_code", type=str)
seismogram_parser.add_argument("station_code", type=str)


class Seismogram(restful.Resource):
    def get(self):
        args = seismogram_parser.parse_args()
        print(args)
        return {'hello': 'world'}


api.add_resource(Seismogram, "/seismogram")


def serve(db_path, port, buffer_size_in_mb):
    app.db = InstaseisDB(db_path=db_path, buffer_size_in_mb=buffer_size_in_mb)
    print(app.db.info)
    app.run(host="0.0.0.0", port=port, debug=True)

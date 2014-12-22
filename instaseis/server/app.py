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
seismogram_parser = restful.reqparse.RequestParser()
# Source parameters.
seismogram_parser.add_argument("source_latitude", type=float, required=True,
                               help="The latitude of the source.")
seismogram_parser.add_argument("source_longitude", type=float, required=True,
                               help="The longitude of the source.")
seismogram_parser.add_argument("source_depth_in_m", type=float,
                               help="The source depth in meter.")
seismogram_parser.add_argument("source_depth_in_m", type=float,
                               help="The source depth in meter.")
# Source can either be given as the moment tensor components in Nm.
seismogram_parser.add_argument("m_rr", type=float,
                               help="rr component of the moment tensor.")
seismogram_parser.add_argument("m_tt", type=float,
                               help="tt component of the moment tensor.")
seismogram_parser.add_argument("m_pp", type=float,
                               help="pp component of the moment tensor.")
seismogram_parser.add_argument("m_rt", type=float,
                               help="rt component of the moment tensor.")
seismogram_parser.add_argument("m_rp", type=float,
                               help="rp component of the moment tensor.")
seismogram_parser.add_argument("m_tp", type=float,
                               help="tp component of the moment tensor.")
# Or as strike, dip, rake and M0.
seismogram_parser.add_argument("strike", type=float,
                               help="strike of the fault in degree")
seismogram_parser.add_argument("dip", type=float,
                               help="dip of the fault in degree")
seismogram_parser.add_argument("rake", type=float,
                               help="rake of the fault in degree")
seismogram_parser.add_argument("M0", type=float,
                               help="scalar seismic moment")
# Or as a force source.
seismogram_parser.add_argument("f_r", type=float,
                               help="r force component in N")
seismogram_parser.add_argument("f_t", type=float,
                               help="t force component in N")
seismogram_parser.add_argument("f_p", type=float,
                               help="p force component in N")
# More optional source parameters.
seismogram_parser.add_argument("source_sliprate", type=float,
                               help="Normalized source time function ("
                                    "sliprate)")
seismogram_parser.add_argument("stf_dt", type=float,
                               help="sampling of the source time function.")
seismogram_parser.add_argument("origin_time", type=str,
                               default=obspy.UTCDateTime(0),
                               help="The origin time of the source. This will "
                               "be the time of the first sample in the final "
                               "seismogram. Be careful to adjust it for any "
                               "time shift or STF (de)convolution effects.")


class Seismogram(restful.Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(Seismogram, "/seismogram")


def serve(db_path, port, buffer_size_in_mb):
    app.db = InstaseisDB(db_path=db_path, buffer_size_in_mb=buffer_size_in_mb)
    print(app.db.info)
    app.run(host="0.0.0.0", port=port, debug=True)

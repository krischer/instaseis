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
from flask import Flask

from ..instaseisdb import InstaseisDB

app = Flask(__name__)


@app.route('/')
def index():
    info = app.db.info
    # No need to write a custom encoder...
    info["datetime"] = str(info["datetime"])
    return flask.jsonify(**info)


def serve(db_path, port, buffer_size_in_mb):
    app.db = InstaseisDB(db_path=db_path, buffer_size_in_mb=buffer_size_in_mb)
    print(app.db.info)
    app.run(host="0.0.0.0", port=port, debug=True)

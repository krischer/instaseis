#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import tornado.web

from ... import __version__


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        response = {
            "type": "Instaseis Remote Server",
            "version": __version__
        }
        self.write(response)
        self.set_header("Access-Control-Allow-Origin", "*")

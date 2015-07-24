#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Instaseis Request handler currently only settings default headers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import tornado

from .. import __version__


class InstaseisRequestHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Server", "InstaseisServer/%s" % __version__)

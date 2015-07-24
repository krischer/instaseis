#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from ..instaseis_request import InstaseisRequestHandler
from ... import __version__


class IndexHandler(InstaseisRequestHandler):
    def get(self):
        response = {
            "type": "Instaseis Remote Server",
            "version": __version__
        }
        self.write(response)

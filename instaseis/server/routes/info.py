#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import copy

from ..instaseis_request import InstaseisRequestHandler


class InfoHandler(InstaseisRequestHandler):
    def get(self):
        info = copy.deepcopy(self.application.db.info)
        # No need to write a custom encoder...
        info["datetime"] = str(info["datetime"])
        info["slip"] = list([float(_i) for _i in info["slip"]])
        info["sliprate"] = list([float(_i) for _i in info["sliprate"]])
        # Clear the directory to avoid leaking any more system information then
        # necessary.
        info["directory"] = ""
        self.write(dict(info))

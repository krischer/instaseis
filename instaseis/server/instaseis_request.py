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
from abc import abstractmethod
import obspy
import tornado

from .. import __version__


class InstaseisRequestHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Server", "InstaseisServer/%s" % __version__)


class InstaseisTimeSeriesHandler(InstaseisRequestHandler):
    arguments = None

    def __init__(self, *args, **kwargs):
        self.__connection_closed = False
        InstaseisRequestHandler.__init__(self, *args, **kwargs)

    def parse_arguments(self):
        # Make sure that no additional arguments are passed.
        unknown_arguments = set(self.request.arguments.keys()).difference(set(
            self.arguments.keys()))
        if unknown_arguments:
            msg = "The following unknown parameters have been passed: %s" % (
                ", ".join("'%s'" % _i for _i in sorted(unknown_arguments)))
            raise tornado.web.HTTPError(400, log_message=msg,
                                        reason=msg)

        # Check for duplicates.
        duplicates = []
        for key, value in self.request.arguments.items():
            if len(value) == 1:
                continue
            else:
                duplicates.append(key)
        if duplicates:
            msg = "Duplicate parameters: %s" % (
                ", ".join("'%s'" % _i for _i in sorted(duplicates)))
            raise tornado.web.HTTPError(400, log_message=msg,
                                        reason=msg)

        args = obspy.core.AttribDict()
        for name, properties in self.arguments.items():
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
                    if "format" in properties:
                        msg = "Parameter '%s' must be formatted as: '%s'" % (
                            name, properties["format"])
                    else:
                        msg = ("Parameter '%s' could not be converted to "
                               "'%s'.") % (
                            name, str(properties["type"].__name__))
                    raise tornado.web.HTTPError(400, log_message=msg,
                                                reason=msg)
            setattr(args, name, value)

        # Validate some of them right here.
        self.validate_parameters(args)

        return args

    @abstractmethod
    def validate_parameters(self, args):
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Server offering a REST API for Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import logging

import tornado.gen
import tornado.ioloop
import tornado.web

from ..instaseis_db import InstaseisDB


application = tornado.web.Application()


from .routes.coordinates import CoordinatesHandler
from .routes.index import IndexHandler
from .routes.info import InfoHandler
from .routes.seismograms import SeismogramsHandler
from .routes.seismograms_raw import RawSeismogramsHandler


application.add_handlers("", [
    (r"/seismograms", SeismogramsHandler),
    (r"/seismograms_raw", RawSeismogramsHandler),
    (r"/info", InfoHandler),
    (r"/", IndexHandler),
    (r"/coordinates", CoordinatesHandler)
])


def launch_io_loop(db_path, port, buffer_size_in_mb, quiet, log_level,
                   station_coordinates_callback=None):
    application.db = InstaseisDB(db_path=db_path,
                                 buffer_size_in_mb=buffer_size_in_mb)
    application.station_coordinates_callback = station_coordinates_callback

    if not quiet:
        # Get all tornado loggers.
        access_log = logging.getLogger("tornado.access")
        app_log = logging.getLogger("tornado.application")
        gen_log = logging.getLogger("tornado.general")
        loggers = (access_log, app_log, gen_log)

        # Console log handler.
        ch = logging.StreamHandler()
        # Add formatter
        FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(FORMAT)
        ch.setFormatter(formatter)

        log_level = getattr(logging, log_level)

        for logger in loggers:
            logger.addHandler(ch)
            logger.setLevel(log_level)

        # Log the database information.
        app_log.info("Successfully opened DB")
        app_log.info(str(application.db))

    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()
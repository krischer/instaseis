#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Server offering a REST API for Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import logging

import tornado.gen
import tornado.ioloop
import tornado.web

from ..database_interfaces import find_and_open_files

from .routes.coordinates import CoordinatesHandler
from .routes.events import EventHandler
from .routes.travel_time import TravelTimeHandler
from .routes.index import IndexHandler
from .routes.info import InfoHandler
from .routes.seismograms import SeismogramsHandler
from .routes.seismograms_raw import RawSeismogramsHandler
from .routes.greens import GreensFunctionHandler
from .routes.finite_source import FiniteSourceSeismogramsHandler


# Bit of a hack: Add geojson to the content-types supported for gzipping.
# The tests will catch if this no longer works with newer tornado versions.
tornado.web.GZipContentEncoding.CONTENT_TYPES.add(
    "application/vnd.geo+json")


def get_application():
    """
    Return the tornado application.

    This is a seperate function to be able to get the same application
    objects for the tests.
    """
    return tornado.web.Application([
        (r"/seismograms", SeismogramsHandler),
        (r"/seismograms_raw", RawSeismogramsHandler),
        (r"/finite_source", FiniteSourceSeismogramsHandler),
        (r"/greens_function", GreensFunctionHandler),
        (r"/info", InfoHandler),
        (r"/", IndexHandler),
        (r"/coordinates", CoordinatesHandler),
        (r"/event", EventHandler),
        (r"/ttimes", TravelTimeHandler)
    ], compress_response=True)


def launch_io_loop(db_path, port, buffer_size_in_mb, quiet, log_level,
                   max_size_of_finite_sources=1000,
                   station_coordinates_callback=None,
                   event_info_callback=None,
                   travel_time_callback=None):  # pragma: no cover
    """
    Launch the instaseis server.

    :param db_path: Path to the database on disc.
    :param port: The desired port of the server.
    :param buffer_size_in_mb: The buffer size in MB per buffer. In most
        cases (which is also the worst case scenario) four buffers will be
        created so over time the maximum memory usage will be four times
        this value.
    :param quiet: Do not log.
    :param log_level: The log level, one of CRITICAL, ERROR, WARNING, INFO,
        DEBUG, NOTSET
    :param max_size_of_finite_sources: The maximum allowed number of point
        sources in a single finite source for the /finite_source route.
    :param station_coordinates_callback: A callback function for station
        coordinates. If not given, certain requests will not be available.
    :param event_info_callback: A callback function returning event
        information. If not given, certain requests will not be available.
    :param travel_time_callback: A callback function returning the travel
        time for certain seismic phase and a given source/receiver geometry.
    """
    application = get_application()
    application.db = find_and_open_files(
        path=db_path, buffer_size_in_mb=buffer_size_in_mb)
    application.station_coordinates_callback = station_coordinates_callback
    application.event_info_callback = event_info_callback

    # This is a callback as currently the instaseis databases don't store
    # the 1D model so we need a way to specify the actually used model. Also
    # gives the option to use other travel time calculation codes.
    application.travel_time_callback = travel_time_callback

    # Maximum number of allowed point sources in the finite source route.
    # Set to None to allow arbitrarily sized finite sources. The calculation
    # might take very long then so be aware!
    application.max_size_of_finite_sources = int(max_size_of_finite_sources)

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

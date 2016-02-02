#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch script for the advanced Instaseis server example.


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import os

from obspy.core.util import geodetics
from obspy.taup import TauPyModel

from instaseis.server.app import launch_io_loop

from station_resolver import parse_station_file, get_coordinates
from event_resolver import get_event_information, create_event_json_file


STATION_FILENAME = "station_list.txt"
conn, cursor = parse_station_file(STATION_FILENAME)

EVENT_FILENAME = "event_db.json"
create_event_json_file(EVENT_FILENAME)


def get_event(event_id):
    return get_event_information(event_id=event_id, filename=EVENT_FILENAME)


def get_station_coordinates(networks, stations):
    return get_coordinates(cursor, networks=networks,
                           stations=stations)


tau_model = TauPyModel(model="ak135")


def get_travel_time(sourcelatitude, sourcelongitude, sourcedepthinmeters,
                    receiverlatitude, receiverlongitude,
                    receiverdepthinmeters, phase_name, db_info):
    if receiverdepthinmeters:
        raise ValueError("This travel time implementation cannot calculate "
                         "buried receivers.")

    great_circle_distance = geodetics.locations2degrees(
        sourcelatitude, sourcelongitude, receiverlatitude, receiverlongitude)

    try:
        tts = tau_model.get_travel_times(
            source_depth_in_km=sourcedepthinmeters / 1000.0,
            distance_in_degree=great_circle_distance,
            phase_list=[phase_name])
    except Exception as e:
        raise ValueError(str(e))

    if not tts:
        return None

    # For any phase, return the first time.
    return tts[0].time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m instaseis.server",
        description='Launch an Instaseis server offering seismograms with a '
                    'REST API.')
    parser.add_argument('--port', type=int, required=True,
                        help='Server port.')
    parser.add_argument('--buffer_size_in_mb', type=int,
                        default=0, help='Size of the buffer in MB')
    parser.add_argument('--max_size_of_finite_sources', type=int,
                        default=1000,
                        help='The maximum allowed number of point sources in '
                             'a single finite source for the /finite_source '
                             'route.')

    parser.add_argument('db_path', type=str,
                        help='Database path')
    parser.add_argument(
        '--quiet', action='store_true',
        help="Don't print any output. Overwrites the 'log_level` setting.")
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        help='The log level for all Tornado loggers.')

    args = parser.parse_args()
    db_path = os.path.abspath(args.db_path)

    launch_io_loop(db_path=db_path, port=args.port,
                   buffer_size_in_mb=args.buffer_size_in_mb,
                   quiet=args.quiet, log_level=args.log_level,
                   max_size_of_finite_sources=args.max_size_of_finite_sources,
                   station_coordinates_callback=get_station_coordinates,
                   event_info_callback=get_event,
                   travel_time_callback=get_travel_time)

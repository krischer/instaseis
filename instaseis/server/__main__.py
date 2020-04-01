#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch Instaseis server.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""  # pragma: no cover
import argparse  # pragma: no cover
import os  # pragma: no cover

from instaseis.server.app import launch_io_loop  # pragma: no cover

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m instaseis.server",
        description="Launch an Instaseis server offering seismograms with a "
        "REST API.",
    )
    parser.add_argument("--port", type=int, required=True, help="Server port.")
    parser.add_argument(
        "--buffer_size_in_mb",
        type=int,
        default=100,
        help="Size of the buffer in MB",
    )
    parser.add_argument(
        "--max_size_of_finite_sources",
        type=int,
        default=1000,
        help="The maximum allowed number of point sources in "
        "a single finite source for the /finite_source "
        "route.",
    )

    parser.add_argument("db_path", type=str, help="Database path")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print any output. Overwrites the 'log_level` setting.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="The log level for all Tornado loggers.",
    )

    args = parser.parse_args()
    db_path = os.path.abspath(args.db_path)

    launch_io_loop(
        db_path=db_path,
        port=args.port,
        buffer_size_in_mb=args.buffer_size_in_mb,
        max_size_of_finite_sources=args.max_size_of_finite_sources,
        quiet=args.quiet,
        log_level=args.log_level,
    )

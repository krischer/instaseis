#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch Instaseis server.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import os

from instaseis.server.app import launch_io_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch an Instaseis server offering seismograms with a '
                    'REST API.')
    parser.add_argument('--port', type=int, required=True,
                        help='Server port.')
    parser.add_argument('--buffer_size_in_mb', type=int,
                        default=100, help='Size of the buffer in MB')
    parser.add_argument('db_path', type=str,
                        help='Database path')

    args = parser.parse_args()
    db_path = os.path.abspath(args.db_path)

    launch_io_loop(db_path=db_path, port=args.port,
                   buffer_size_in_mb=args.buffer_size_in_mb)

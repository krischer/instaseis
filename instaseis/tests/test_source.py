#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for source handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import obspy
import os

from instaseis import Source

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVENT_FILE = os.path.join(DATA, "GCMT_event_STRAIT_OF_GIBRALTAR.xml")


# def test_parse_STATIONS_file(tmpdir):
#     """
#     Tests parsing from a STATIONS file. tmpdir is a pytest fixture.
#     """
#     filename = os.path.join(tmpdir.dirname, "STATIONS")
#     lines = (
#         "AAK        II       10.     20.   1645.0    30.0",
#         "BBK        AA       20.     30.   1645.0    30.0"
#     )
#     with open(filename, "wt") as fh:
#         fh.write("\n".join(lines))
#
#     receivers = Receiver.parse(filename)
#
#     assert len(receivers) == 2
#
#     rec = receivers[0]
#     assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
#            (10.0, 20.0, "II", "AAK")
#
#     rec = receivers[1]
#     assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
#            (20.0, 30.0, "AA", "BBK")

def _assert_src(src):
    """
    We constantly test the same event in various configurations.
    """
    assert (src.latitude, src.longitude, src.depth_in_m, src.m_rr, src.m_tt,
            src.m_pp, src.m_rt, src.m_rp, src.m_tp) == \
           (36.97, -3.54, 609800.0, -2.16E18, 5.36E17, 1.62E18, 1.3E16,
            3.23E18, 1.75E18)


def test_parse_QuakeML():
    """
    Tests parsing from a QuakeML file.
    """
    src = Source.parse(EVENT_FILE)
    _assert_src(src)


def test_parse_obspy_objects():
    """
    Tests parsing from ObsPy objects.
    """
    cat = obspy.readEvents(EVENT_FILE)
    ev = cat[0]

    _assert_src(Source.parse(cat))
    _assert_src(Source.parse(ev))

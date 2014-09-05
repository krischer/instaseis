#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the receiver handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import obspy
import os

from ..source import Receiver

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_parse_STATIONS_file(tmpdir):
    """
    Tests parsing from a STATIONS file. tmpdir is a pytest fixture.
    """
    filename = os.path.join(tmpdir.dirname, "STATIONS")
    lines = (
        "AAK        II       10.     20.   1645.0    30.0",
        "BBK        AA       20.     30.   1645.0    30.0"
    )
    with open(filename, "wt") as fh:
        fh.write("\n".join(lines))

    receivers = Receiver.parse(filename)

    assert len(receivers) == 2

    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (10.0, 20.0, "II", "AAK")

    rec = receivers[1]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (20.0, 30.0, "AA", "BBK")


def test_parse_StationXML():
    filename = os.path.join(DATA, "TA.Q56A..BH.xml")
    receivers = Receiver.parse(filename)

    assert len(receivers) == 1
    rec = receivers[0]

    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
       (39.041, -79.1871, "TA", "Q56A")


def test_parse_obspy_objects():
    filename = os.path.join(DATA, "TA.Q56A..BH.xml")
    inv = obspy.read_inventory(filename)

    receivers = Receiver.parse(inv)
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (39.041, -79.1871, "TA", "Q56A")

    receivers = Receiver.parse(inv[0])
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (39.041, -79.1871, "TA", "Q56A")

    receivers = Receiver.parse(inv[0][0], network_code="TA")
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (39.041, -79.1871, "TA", "Q56A")

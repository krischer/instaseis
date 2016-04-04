#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the receiver handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import obspy
import os
import pytest

from instaseis import Receiver, ReceiverParseError
from instaseis.helpers import elliptic_to_geocentric_latitude

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
           (elliptic_to_geocentric_latitude(10.0), 20.0, "II", "AAK")

    rec = receivers[1]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (elliptic_to_geocentric_latitude(20.0), 30.0, "AA", "BBK")


def test_parse_StationXML():
    filename = os.path.join(DATA, "TA.Q56A..BH.xml")
    receivers = Receiver.parse(filename)

    assert len(receivers) == 1
    rec = receivers[0]

    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
        (elliptic_to_geocentric_latitude(39.041), -79.1871, "TA", "Q56A")


def test_parse_obspy_objects():
    filename = os.path.join(DATA, "TA.Q56A..BH.xml")
    inv = obspy.read_inventory(filename)

    receivers = Receiver.parse(inv)
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (elliptic_to_geocentric_latitude(39.041), -79.1871, "TA", "Q56A")

    receivers = Receiver.parse(inv[0])
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (elliptic_to_geocentric_latitude(39.041), -79.1871, "TA", "Q56A")

    receivers = Receiver.parse(inv[0][0], network_code="TA")
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == \
           (elliptic_to_geocentric_latitude(39.041), -79.1871, "TA", "Q56A")


def test_parse_sac_files():
    filename = os.path.join(DATA, "example.sac")
    receivers = Receiver.parse(filename)

    assert len(receivers) == 1
    rec = receivers[0]
    assert (round(rec.latitude, 3), round(rec.longitude, 3),
            rec.network, rec.station) == \
        (round(elliptic_to_geocentric_latitude(34.94598), 3),
         round(-106.45713, 3), 'IU', 'ANMO')


def test_parse_sac_file_without_coordinates():
    filename = os.path.join(DATA, "example_without_coordinates.sac")
    with pytest.raises(ReceiverParseError) as e:
        Receiver.parse(filename)

    assert "SAC file does not contain coordinates for channel".lower() in \
        str(e).lower()


def test_parse_obspy_waveform_objects():
    filename = os.path.join(DATA, "example.sac")
    st = obspy.read(filename)

    # From stream.
    receivers = Receiver.parse(st)
    assert len(receivers) == 1
    rec = receivers[0]
    # Coordinates are assumed to be WGS84 and will be converted to geocentric.
    assert (round(rec.latitude, 3), round(rec.longitude, 3),
            rec.network, rec.station) == \
           (round(elliptic_to_geocentric_latitude(34.94598), 3),
            round(-106.45713, 3), 'IU', 'ANMO')

    # From trace.
    receivers = Receiver.parse(st[0])
    assert len(receivers) == 1
    rec = receivers[0]
    assert (round(rec.latitude, 3), round(rec.longitude, 3),
            rec.network, rec.station) == \
           (round(elliptic_to_geocentric_latitude(34.94598), 3),
            round(-106.45713, 3), 'IU', 'ANMO')


def test_duplicate_receivers():
    """
    Many waveform files contain multiple channels of the same stations. Of
    course these duplicates need to be purged.
    """
    filename = os.path.join(DATA, "example.sac")
    st = obspy.read(filename)
    st += st.copy()
    st[1].stats.channel = "LHZ"

    receivers = Receiver.parse(st)
    assert len(receivers) == 1
    rec = receivers[0]
    # Coordinates are assumed to be WGS84 and will be converted to geocentric.
    assert (round(rec.latitude, 3), round(rec.longitude, 3),
            rec.network, rec.station) == \
           (round(elliptic_to_geocentric_latitude(34.94598), 3),
            round(-106.45713, 3), 'IU', 'ANMO')


def test_dataless_seed_files():
    filename = os.path.join(DATA, "dataless.seed.BW_FURT")
    receivers = Receiver.parse(filename)
    assert len(receivers) == 1
    rec = receivers[0]
    # Coordinates are assumed to be WGS84 and will be converted to geocentric.
    assert (round(rec.latitude, 3), round(rec.longitude, 3),
            rec.network, rec.station) == \
           (round(elliptic_to_geocentric_latitude(48.162899), 3),
            round(11.2752, 3), 'BW', 'FURT')


def test_station_x_y_z():
    station = Receiver(latitude=42.6390, longitude=74.4940, depth_in_m=0.0)
    assert abs(station.x() - 1252949.21995) < 1E-5
    assert abs(station.y() - 4516152.38916) < 1E-5
    assert abs(station.z() - 4315567.96379) < 1E-5
    assert abs(station.colatitude - 47.3609999) < 1E-5
    assert station.depth_in_m == 0.0
    assert station.radius_in_m() == 6371000.0


def test_str_method_of_receiver():
    """
    Tests the string method of the receiver class.
    """
    rec = Receiver(latitude=1.0, longitude=2.0, network="BW", station="ALTM")
    assert str(rec) == (
        "Instaseis Receiver:\n"
        "\tlongitude :    2.0 deg\n"
        "\tlatitude  :    1.0 deg\n"
        "\tnetwork   : BW\n"
        "\tstation   : ALTM\n"
        "\tlocation  : \n"
    )

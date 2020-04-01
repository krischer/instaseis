#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the receiver handling.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import io
import obspy
import os
import pytest

from instaseis import Receiver, ReceiverParseError
from instaseis.helpers import elliptic_to_geocentric_latitude

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_parse_stations_file(tmpdir):
    """
    Tests parsing from a STATIONS file. tmpdir is a pytest fixture.
    """
    filename = os.path.join(tmpdir.dirname, "STATIONS")
    lines = (
        "AAK        II       10.     20.   1645.0    30.0",
        "BBK        AA       20.     30.   1645.0    30.0",
    )
    with open(filename, "wt") as fh:
        fh.write("\n".join(lines))

    receivers = Receiver.parse(filename)

    assert len(receivers) == 2

    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == (
        elliptic_to_geocentric_latitude(10.0),
        20.0,
        "II",
        "AAK",
    )

    rec = receivers[1]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == (
        elliptic_to_geocentric_latitude(20.0),
        30.0,
        "AA",
        "BBK",
    )


def test_parse_stationxml():
    filename = os.path.join(DATA, "TA.Q56A..BH.xml")
    receivers = Receiver.parse(filename)

    assert len(receivers) == 1
    rec = receivers[0]

    assert (rec.latitude, rec.longitude, rec.network, rec.station) == (
        elliptic_to_geocentric_latitude(39.041),
        -79.1871,
        "TA",
        "Q56A",
    )


def test_parse_obspy_objects():
    filename = os.path.join(DATA, "TA.Q56A..BH.xml")
    inv = obspy.read_inventory(filename)

    receivers = Receiver.parse(inv)
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == (
        elliptic_to_geocentric_latitude(39.041),
        -79.1871,
        "TA",
        "Q56A",
    )

    receivers = Receiver.parse(inv[0])
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == (
        elliptic_to_geocentric_latitude(39.041),
        -79.1871,
        "TA",
        "Q56A",
    )

    receivers = Receiver.parse(inv[0][0], network_code="TA")
    assert len(receivers) == 1
    rec = receivers[0]
    assert (rec.latitude, rec.longitude, rec.network, rec.station) == (
        elliptic_to_geocentric_latitude(39.041),
        -79.1871,
        "TA",
        "Q56A",
    )


def test_parse_sac_files():
    filename = os.path.join(DATA, "example.sac")
    receivers = Receiver.parse(filename)

    assert len(receivers) == 1
    rec = receivers[0]
    assert (
        round(rec.latitude, 3),
        round(rec.longitude, 3),
        rec.network,
        rec.station,
    ) == (
        round(elliptic_to_geocentric_latitude(34.94598), 3),
        round(-106.45713, 3),
        "IU",
        "ANMO",
    )


def test_parse_sac_file_without_coordinates():
    filename = os.path.join(DATA, "example_without_coordinates.sac")
    with pytest.raises(ReceiverParseError) as e:
        Receiver.parse(filename)

    assert (
        "SAC file does not contain coordinates for channel".lower()
        in str(e).lower()
    )


def test_parse_obspy_waveform_objects():
    filename = os.path.join(DATA, "example.sac")
    st = obspy.read(filename)

    # From stream.
    receivers = Receiver.parse(st)
    assert len(receivers) == 1
    rec = receivers[0]
    # Coordinates are assumed to be WGS84 and will be converted to geocentric.
    assert (
        round(rec.latitude, 3),
        round(rec.longitude, 3),
        rec.network,
        rec.station,
    ) == (
        round(elliptic_to_geocentric_latitude(34.94598), 3),
        round(-106.45713, 3),
        "IU",
        "ANMO",
    )

    # From trace.
    receivers = Receiver.parse(st[0])
    assert len(receivers) == 1
    rec = receivers[0]
    assert (
        round(rec.latitude, 3),
        round(rec.longitude, 3),
        rec.network,
        rec.station,
    ) == (
        round(elliptic_to_geocentric_latitude(34.94598), 3),
        round(-106.45713, 3),
        "IU",
        "ANMO",
    )


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
    assert (
        round(rec.latitude, 3),
        round(rec.longitude, 3),
        rec.network,
        rec.station,
    ) == (
        round(elliptic_to_geocentric_latitude(34.94598), 3),
        round(-106.45713, 3),
        "IU",
        "ANMO",
    )


def test_dataless_seed_files():
    filename = os.path.join(DATA, "dataless.seed.BW_FURT")
    receivers = Receiver.parse(filename)
    assert len(receivers) == 1
    rec = receivers[0]
    # Coordinates are assumed to be WGS84 and will be converted to geocentric.
    assert (
        round(rec.latitude, 3),
        round(rec.longitude, 3),
        rec.network,
        rec.station,
    ) == (
        round(elliptic_to_geocentric_latitude(48.162899), 3),
        round(11.2752, 3),
        "BW",
        "FURT",
    )


def test_station_x_y_z():
    station = Receiver(latitude=42.6390, longitude=74.4940, depth_in_m=0.0)
    assert abs(station.x() - 1252949.21995) < 1e-5
    assert abs(station.y() - 4516152.38916) < 1e-5
    assert abs(station.z() - 4315567.96379) < 1e-5
    assert abs(station.colatitude - 47.3609999) < 1e-5
    assert station.depth_in_m == 0.0
    assert station.radius_in_m() == 6371000.0


def test_str_method_of_receiver():
    """
    Tests the string method of the receiver class.
    """
    rec = Receiver(latitude=1.0, longitude=2.0, network="BW", station="ALTM")
    assert str(rec) == (
        "Instaseis Receiver:\n"
        "\tLongitude :    2.0 deg\n"
        "\tLatitude  :    1.0 deg\n"
        "\tNetwork   : BW\n"
        "\tStation   : ALTM\n"
        "\tLocation  : \n"
    )


def test_error_handling_when_parsing_station_files(tmpdir):
    """
    Tests error handling when parsing station files.
    """
    # Differing coordinates for channels of the same station.
    inv = obspy.read_inventory()
    with pytest.raises(ReceiverParseError) as err:
        inv[0][0][0].latitude -= 10
        Receiver.parse(inv)
    assert err.value.args[0] == (
        "The coordinates of the channels of station "
        "'GR.FUR' are not identical."
    )

    # Once again, with a file.
    with io.BytesIO() as buf:
        inv.write(buf, format="stationxml")
        buf.seek(0)
        with pytest.raises(ReceiverParseError) as err:
            Receiver.parse(buf)
    assert err.value.args[0] == (
        "The coordinates of the channels of station "
        "'GR.FUR' are not identical."
    )

    # ObsPy Trace without a sac attribute.
    with pytest.raises(ReceiverParseError) as err:
        Receiver.parse(obspy.read())
    assert err.value.args[0] == ("ObsPy Trace must have an sac attribute.")

    # Trigger error when a SEED files has differing origins.
    filename = os.path.join(DATA, "dataless.seed.BW_FURT")
    p = obspy.io.xseed.parser.Parser(filename)
    p.blockettes[52][1].latitude += 1
    with pytest.raises(ReceiverParseError) as err:
        Receiver.parse(p)
    assert err.value.args[0] == (
        "The coordinates of the channels of station "
        "'BW.FURT' are not identical."
    )

    # Same thing but this time with a file.
    tmpfile = os.path.join(tmpdir.strpath, "temp.seed")
    p.write_seed(tmpfile)
    with pytest.raises(ReceiverParseError) as err:
        Receiver.parse(tmpfile)
    assert err.value.args[0] == (
        "The coordinates of the channels of station "
        "'BW.FURT' are not identical."
    )

    # Parsing random string.
    with pytest.raises(ValueError) as err:
        Receiver.parse("random_string")
    assert err.value.args[0] == "'random_string' could not be parsed."


def test_invalid_lat_lng_values():
    """
    Tests invalid latitude/longitude values
    """
    Receiver(latitude=10, longitude=10)

    with pytest.raises(ValueError):
        Receiver(latitude=100, longitude=10)

    with pytest.raises(ValueError):
        Receiver(latitude=-100, longitude=10)

    with pytest.raises(ValueError):
        Receiver(latitude=10, longitude=200)

    with pytest.raises(ValueError):
        Receiver(latitude=10, longitude=-200)

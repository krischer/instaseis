#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Instaseis server.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import copy
import json
import obspy
import numpy as np
import instaseis
from .tornado_testing_fixtures import *  # NOQA

# Conditionally import mock either from the stdlib or as a separate library.
import sys
if sys.version_info[0] == 2:
    import mock
else:
    import unittest.mock as mock


def _assemble_url(**kwargs):
    """
    Helper function.
    """
    url = "/seismograms?"
    url += "&".join("%s=%s" % (key, value) for key, value in kwargs.items())
    return url


def _assemble_url_raw(**kwargs):
    """
    Helper function.
    """
    url = "/seismograms_raw?"
    url += "&".join("%s=%s" % (key, value) for key, value in kwargs.items())
    return url


def test_root_route(all_clients):
    """
    Shows very basic information and the version of the client. Test is run
    for all clients.
    """
    client = all_clients
    request = client.fetch("/")
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert result == {
        "type": "Instaseis Remote Server", "version": instaseis.__version__}
    assert request.headers["Content-Type"] == "application/json; charset=UTF-8"


def test_info_route(all_clients):
    """
    Tests that the /info route returns the information dictionary and does
    not mess with anything.

    Test is parameterized to run for all test databases.
    """
    client = all_clients
    # Load the result via the webclient.
    request = client.fetch("/info")
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/json; charset=UTF-8"
    result = json.loads(str(request.body.decode("utf8")))
    # Convert list to arrays.
    client_slip = np.array(result["slip"])
    client_sliprate = np.array(result["sliprate"])
    del result["slip"]
    del result["sliprate"]

    # Make sure it is identical to one from a local client.
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    db_slip = list(db.info.slip)
    db_sliprate = list(db.info.sliprate)
    del db.info["slip"]
    del db.info["sliprate"]

    # Directory from server is empty.
    db.info["directory"] = ""
    # Datetime is a string.
    db.info["datetime"] = str(db.info["datetime"])

    # Make sure the information dictionary is the same, no matter where it
    # comes from.
    assert dict(db.info) == result
    # Same for slip and sliprate.
    np.testing.assert_allclose(client_slip, db_slip)
    np.testing.assert_allclose(client_sliprate, db_sliprate)


def test_raw_seismograms_error_handling(all_clients):
    """
    Tests error handling of the /seismograms_raw route. Potentially outwards
    facing thus tested rather well.
    """
    client = all_clients

    basic_parameters = {
        "source_latitude": 10,
        "source_longitude": 10,
        "receiver_latitude": -10,
        "receiver_longitude": -10}

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["source_latitude"]
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'source_latitude' not given."

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["source_latitude"] = "A"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "source_latitude" in request.reason

    # No source.
    request = client.fetch(_assemble_url_raw(**basic_parameters))
    assert request.code == 400
    assert request.reason == "No/insufficient source parameters specified"

    # Invalid receiver.
    params = copy.deepcopy(basic_parameters)
    params["receiver_latitude"] = "100"
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct receiver with " in request.reason.lower()

    # Invalid MT source.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["strike"] = "45"
    params["dip"] = "45"
    params["rake"] = "45"
    params["M0"] = "450000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct the source" in request.reason.lower()
    assert "strike/dip/rake" in request.reason.lower()

    # Invalid force source. It only works in displ_only mode but here it
    # fails earlier.
    params = copy.deepcopy(basic_parameters)
    params["f_r"] = "100000"
    params["f_t"] = "100000"
    params["f_p"] = "100000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct force source" in request.reason.lower()

    # Could not extract seismogram.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["components"] = "ABC"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()

    # Unlikely to be raised for real, but test the resulting error nonetheless.
    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    "._convert_to_stream") as p:
        p.side_effect = Exception

        params = copy.deepcopy(basic_parameters)
        params["m_tt"] = "100000"
        params["m_pp"] = "100000"
        params["m_rr"] = "100000"
        params["m_rt"] = "100000"
        params["m_rp"] = "100000"
        params["m_tp"] = "100000"
        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 500
        assert "could not convert seismogram to a" in request.reason.lower()


def test_seismograms_raw_route(all_clients):
    """
    Test the raw routes. Make sure the response is a MiniSEED file with the
    correct channels.

    Once again executed for each known test database.
    """
    client = all_clients

    basic_parameters = {
        "source_latitude": 10,
        "source_longitude": 10,
        "receiver_latitude": -10,
        "receiver_longitude": -10}

    # Various sources.
    mt = {"m_tt": "100000", "m_pp": "100000", "m_rr": "100000",
          "m_rt": "100000", "m_rp": "100000", "m_tp": "100000"}
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    fs = {"f_r": "100000", "f_t": "100000", "f_p": "100000"}

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == 3

    # Assert the MiniSEED file and some basic properties.
    for tr in st:
        assert hasattr(tr.stats, "mseed")
        assert tr.data.dtype.char == "f"

    # Strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params.update(sdr)
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == 3

    # Assert the MiniSEED file and some basic properties.
    for tr in st:
        assert hasattr(tr.stats, "mseed")
        assert tr.data.dtype.char == "f"

    # Force source only works for displ_only databases.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params.update(fs)
        time = obspy.UTCDateTime(2008, 7, 6, 5, 4, 3)
        params["origin_time"] = str(time)
        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        st = obspy.read(request.buffer)
        assert len(st) == 3

        # Assert the MiniSEED file and some basic properties.
        for tr in st:
            assert hasattr(tr.stats, "mseed")
            assert tr.data.dtype.char == "f"
            assert tr.stats.starttime == time

    # Test different components.
    components = ["NRE", "ZRT", "RT", "Z", "ZNE"]
    for comp in components:
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["components"] = comp
        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        st = obspy.read(request.buffer)
        assert len(st) == len(comp)
        assert "".join(sorted(comp)) == "".join(sorted(
            [tr.stats.channel[-1] for tr in st]))

    # Test passing the origin time.
    params = copy.deepcopy(basic_parameters)
    time = obspy.UTCDateTime(2013, 1, 2, 3, 4, 5)
    params.update(mt)
    params["origin_time"] = str(time)
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == 3
    for tr in st:
        assert tr.stats.starttime == time

    # Test passing network and station codes.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["network_code"] = "BW"
    params["station_code"] = "ALTM"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == 3
    for tr in st:
        assert tr.stats.network == "BW"
        assert tr.stats.station == "ALTM"


def test_mu_is_passed_as_header_value(all_clients):
    """
    Makes sure mu is passed as a header value.

    Also tests the other headers.
    """
    client = all_clients
    parameters = {"source_latitude": 10, "source_longitude": 10,
                  "receiver_latitude": -10, "receiver_longitude": -10,
                  "m_tt": "100000", "m_pp": "100000", "m_rr": "100000",
                  "m_rt": "100000", "m_rp": "100000", "m_tp": "100000"}

    # Moment tensor source.
    request = client.fetch(_assemble_url_raw(**parameters))
    assert request.code == 200
    # Make sure the mu header exists and the value can be converted to a float.
    assert "Instaseis-Mu" in request.headers
    assert isinstance(float(request.headers["Instaseis-Mu"]), float)

    assert request.headers["Content-Type"] == "application/octet-stream"
    cd = request.headers["Content-Disposition"]
    assert "attachment; filename=" in cd
    assert "instaseis_seismogram" in cd


def test_object_creation_for_raw_seismogram_route(all_clients):
    """
    Tests that the correct objects are created for the raw seismogram route.
    """
    client = all_clients

    basic_parameters = {
        "source_latitude": 10,
        "source_longitude": 10,
        "receiver_latitude": -10,
        "receiver_longitude": -10}

    # Various sources.
    mt = {"m_tt": "100000", "m_pp": "100000", "m_rr": "100000",
          "m_rt": "100000", "m_rp": "100000", "m_tp": "100000"}
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    fs = {"f_r": "100000", "f_t": "100000", "f_p": "100000"}

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    with mock.patch("instaseis.instaseis_db.InstaseisDB._get_seismograms") \
            as p:
        _st = obspy.read()
        data = {}
        data["mu"] = 1.0
        for tr in _st:
            data[tr.stats.channel[-1]] = tr.data
        p.return_value = data

        # Moment tensor source.
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["source_latitude"],
            longitude=basic_parameters["source_longitude"],
            depth_in_m=0.0,
            **dict((key, float(value)) for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=0.0)

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["source_depth_in_m"] = "5.0"
        params["origin_time"] = str(time)
        params["receiver_depth_in_m"] = "55.0"
        params["network_code"] = "BW"
        params["station_code"] = "ALTM"

        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["source_latitude"],
            longitude=basic_parameters["source_longitude"],
            depth_in_m=5.0, origin_time=time,
            **dict((key, float(value)) for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=55.0, network="BW", station="ALTM")

        # From strike, dip, rake
        p.reset_mock()

        params = copy.deepcopy(basic_parameters)
        params.update(sdr)
        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=0.0,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=0.0)

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["source_depth_in_m"] = "5.0"
        params["origin_time"] = str(time)
        params["receiver_depth_in_m"] = "55.0"
        params["network_code"] = "BW"
        params["station_code"] = "ALTM"

        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=5.0, origin_time=time,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=55.0, network="BW", station="ALTM")

        # Force source only works for displ_only databases.
        if "displ_only" in client.filepath:
            p.reset_mock()

            params = copy.deepcopy(basic_parameters)
            params.update(fs)
            request = client.fetch(_assemble_url_raw(**params))
            assert request.code == 200

            assert p.call_count == 1
            assert p.call_args[1]["components"] == ["Z", "N", "E"]
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=0.0,
                **dict((key, float(value)) for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiver_latitude"],
                longitude=basic_parameters["receiver_longitude"],
                depth_in_m=0.0)

            # Moment tensor source with a couple more parameters.
            p.reset_mock()

            params["source_depth_in_m"] = "5.0"
            params["origin_time"] = str(time)
            params["receiver_depth_in_m"] = "55.0"
            params["network_code"] = "BW"
            params["station_code"] = "ALTM"

            request = client.fetch(_assemble_url_raw(**params))
            assert request.code == 200

            assert p.call_count == 1
            assert p.call_args[1]["components"] == ["Z", "N", "E"]
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=5.0, origin_time=time,
                **dict((key, float(value)) for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiver_latitude"],
                longitude=basic_parameters["receiver_longitude"],
                depth_in_m=55.0, network="BW", station="ALTM")


def test_seismograms_error_handling(all_clients):
    """
    Tests error handling of the /seismograms route. Potentially outwards
    facing thus tested rather well.
    """
    client = all_clients

    basic_parameters = {
        "source_latitude": 10,
        "source_longitude": 10,
        "receiver_latitude": -10,
        "receiver_longitude": -10}

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["source_latitude"]
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'source_latitude' not given."

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["source_latitude"] = "A"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "source_latitude" in request.reason

    # No source.
    request = client.fetch(_assemble_url(**basic_parameters))
    assert request.code == 400
    assert request.reason == "No/insufficient source parameters specified"

    # Invalid receiver.
    params = copy.deepcopy(basic_parameters)
    params["receiver_latitude"] = "100"
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct receiver with " in request.reason.lower()

    # Invalid MT source.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["strike"] = "45"
    params["dip"] = "45"
    params["rake"] = "45"
    params["M0"] = "450000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct the source" in request.reason.lower()
    assert "strike/dip/rake" in request.reason.lower()

    # Invalid force source. It only works in displ_only mode but here it
    # fails earlier.
    params = copy.deepcopy(basic_parameters)
    params["f_r"] = "100000"
    params["f_t"] = "100000"
    params["f_p"] = "100000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct force source" in request.reason.lower()

    # Could not extract seismogram.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["components"] = "ABC"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()

    # Wrong type of seismogram requested.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["kind"] = "fun"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "kind must be one of" in request.reason.lower()

    # dt is too small - protects the server from having to serve humongous
    # files.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["dt"] = "0.009"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "the smallest possible dt is 0.01" in request.reason.lower()

    # lanzcos window is too wide or too narrow.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["a_lanczos"] = "1"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "`a_lanczos` must not be smaller" in request.reason.lower()
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["a_lanczos"] = "21"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "`a_lanczos` must not be smaller" in request.reason.lower()


def test_conversion_to_boolean_parameters(all_clients):
    """
    Boolean values can be specified in a number of ways. Test that these are
    working as expected.
    """
    client = all_clients

    basic_parameters = {
        "source_latitude": 10,
        "source_longitude": 10,
        "receiver_latitude": -10,
        "receiver_longitude": -10,
        "m_tt": "100000",
        "m_pp": "100000",
        "m_rr": "100000",
        "m_rt": "100000",
        "m_rp": "100000",
        "m_tp": "100000"}

    truth_values = ["1", "true", "TRUE", "True", "T", "t", "y", "Y"]
    false_values = ["0", "false", "FALSE", "False", "F", "f", "n", "N"]
    invalid_values = ["A", "HMMMM", "234"]

    for value in truth_values:
        params = copy.deepcopy(basic_parameters)
        params["remove_source_shift"] = value
        _st = obspy.read()

        with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                        ".get_seismograms") as p:
            p.return_value = _st
            client.fetch(_assemble_url(**params))
            assert p.call_count == 1
            assert p.call_args[1]["remove_source_shift"] is True

    for value in false_values:
        params = copy.deepcopy(basic_parameters)
        params["remove_source_shift"] = value
        _st = obspy.read()

        with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                        ".get_seismograms") as p:
            p.return_value = _st
            client.fetch(_assemble_url(**params))
            assert p.call_count == 1
            assert p.call_args[1]["remove_source_shift"] is False

    # Test invalid values.
    for value in invalid_values:
        params = copy.deepcopy(basic_parameters)
        params["remove_source_shift"] = value
        request = client.fetch(_assemble_url(**params))
        assert request.code == 400
        assert ("parameter 'remove_source_shift' could not be converted to "
                "'bool'" in request.reason.lower())


def test_object_creation_for_seismogram_route(all_clients):
    """
    Tests that the correct objects are created for the seismogram route.
    """
    client = all_clients

    basic_parameters = {
        "source_latitude": 10,
        "source_longitude": 10,
        "receiver_latitude": -10,
        "receiver_longitude": -10}

    # Various sources.
    mt = {"m_tt": "100000", "m_pp": "100000", "m_rr": "100000",
          "m_rt": "100000", "m_rp": "100000", "m_tp": "100000"}
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    fs = {"f_r": "100000", "f_t": "100000", "f_p": "100000"}

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    with mock.patch("instaseis.instaseis_db.InstaseisDB.get_seismograms") \
            as p:
        _st = obspy.read()
        p.return_value = _st

        # Moment tensor source.
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["source_latitude"],
            longitude=basic_parameters["source_longitude"],
            depth_in_m=0.0,
            **dict((key, float(value)) for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=0.0)
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["source_depth_in_m"] = "5.0"
        params["origin_time"] = str(time)
        params["receiver_depth_in_m"] = "55.0"
        params["network_code"] = "BW"
        params["station_code"] = "ALTM"

        request = client.fetch(_assemble_url(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["source_latitude"],
            longitude=basic_parameters["source_longitude"],
            depth_in_m=5.0, origin_time=time,
            **dict((key, float(value)) for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=55.0, network="BW", station="ALTM")
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        # From strike, dip, rake
        p.reset_mock()

        params = copy.deepcopy(basic_parameters)
        params.update(sdr)
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=0.0,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=0.0)
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["source_depth_in_m"] = "5.0"
        params["origin_time"] = str(time)
        params["receiver_depth_in_m"] = "55.0"
        params["network_code"] = "BW"
        params["station_code"] = "ALTM"

        request = client.fetch(_assemble_url(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=5.0, origin_time=time,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiver_latitude"],
            longitude=basic_parameters["receiver_longitude"],
            depth_in_m=55.0, network="BW", station="ALTM")
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        # Force source only works for displ_only databases.
        if "displ_only" in client.filepath:
            p.reset_mock()

            params = copy.deepcopy(basic_parameters)
            params.update(fs)
            request = client.fetch(_assemble_url(**params))
            assert request.code == 200

            assert p.call_count == 1
            assert p.call_args[1]["components"] == ["Z", "N", "E"]
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=0.0,
                **dict((key, float(value)) for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiver_latitude"],
                longitude=basic_parameters["receiver_longitude"],
                depth_in_m=0.0)
            assert p.call_args[1]["kind"] == "displacement"
            assert p.call_args[1]["remove_source_shift"] is True
            assert p.call_args[1]["reconvolve_stf"] is False
            assert p.call_args[1]["return_obspy_stream"] is True
            assert p.call_args[1]["dt"] is None
            assert p.call_args[1]["a_lanczos"] == 5

            # Moment tensor source with a couple more parameters.
            p.reset_mock()

            params["source_depth_in_m"] = "5.0"
            params["origin_time"] = str(time)
            params["receiver_depth_in_m"] = "55.0"
            params["network_code"] = "BW"
            params["station_code"] = "ALTM"

            request = client.fetch(_assemble_url(**params))
            assert request.code == 200

            assert p.call_count == 1
            assert p.call_args[1]["components"] == ["Z", "N", "E"]
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["source_latitude"],
                longitude=basic_parameters["source_longitude"],
                depth_in_m=5.0, origin_time=time,
                **dict((key, float(value)) for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiver_latitude"],
                longitude=basic_parameters["receiver_longitude"],
                depth_in_m=55.0, network="BW", station="ALTM")
            assert p.call_args[1]["kind"] == "displacement"
            assert p.call_args[1]["remove_source_shift"] is True
            assert p.call_args[1]["reconvolve_stf"] is False
            assert p.call_args[1]["return_obspy_stream"] is True
            assert p.call_args[1]["dt"] is None
            assert p.call_args[1]["a_lanczos"] == 5

        # Now test other the other parameters.
        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["components"] = "RTE"
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["R", "T", "E"]
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["kind"] = "acceleration"
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "acceleration"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["kind"] = "velocity"
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "velocity"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["kind"] = "VeLoCity"
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "velocity"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["remove_source_shift"] = "False"
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["dt"] = "0.1"
        params["a_lanczos"] = "20"
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] == 0.1
        assert p.call_args[1]["a_lanczos"] == 20

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["dt"] = "0.1"
        params["a_lanczos"] = "2"
        params["kind"] = "ACCELERATION"
        params["remove_source_shift"] = "False"
        request = client.fetch(_assemble_url(**params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "acceleration"
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] == 0.1
        assert p.call_args[1]["a_lanczos"] == 2

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Instaseis server.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import copy
import io
import json
import zipfile

import obspy
import numpy as np
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
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcelatitude"]
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'sourcelatitude' not given."

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["sourcelatitude"] = "A"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "sourcelatitude" in request.reason

    # No source.
    request = client.fetch(_assemble_url_raw(**basic_parameters))
    assert request.code == 400
    assert request.reason == "No/insufficient source parameters specified"

    # Invalid receiver.
    params = copy.deepcopy(basic_parameters)
    params["receiverlatitude"] = "100"
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct receiver with " in request.reason.lower()

    # Invalid MT source.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["strike"] = "45"
    params["dip"] = "45"
    params["rake"] = "45"
    params["M0"] = "450000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct the source" in request.reason.lower()
    assert "strike/dip/rake" in request.reason.lower()

    # Invalid force source. It only works in displ_only mode but here it
    # fails earlier.
    params = copy.deepcopy(basic_parameters)
    params["fr"] = "100000"
    params["ft"] = "100000"
    params["fp"] = "100000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not construct force source" in request.reason.lower()

    # Could not extract seismogram.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["components"] = "ABC"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()

    # Unlikely to be raised for real, but test the resulting error nonetheless.
    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    "._convert_to_stream") as p:
        p.side_effect = Exception

        params = copy.deepcopy(basic_parameters)
        params["mtt"] = "100000"
        params["mpp"] = "100000"
        params["mrr"] = "100000"
        params["mrt"] = "100000"
        params["mrp"] = "100000"
        params["mtp"] = "100000"
        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 500
        assert "could not convert seismogram to a" in request.reason.lower()

    # too many components raise to avoid abuse.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["components"] = "NNEERRTTZZ"
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "a maximum of 5 components can be request" in request.reason.lower()

    # At least one components must be requested.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["components"] = ""
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 400
    assert "a request with no components will not re" in request.reason.lower()


def test_seismograms_raw_route(all_clients):
    """
    Test the raw routes. Make sure the response is a MiniSEED file with the
    correct channels.

    Once again executed for each known test database.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    # Various sources.
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "100000", "fp": "100000"}

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
        params["origintime"] = str(time)
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
    params["origintime"] = str(time)
    request = client.fetch(_assemble_url_raw(**params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == 3
    for tr in st:
        assert tr.stats.starttime == time

    # Test passing network and station codes.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"
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
    parameters = {"sourcelatitude": 10, "sourcelongitude": 10,
                  "receiverlatitude": -10, "receiverlongitude": -10,
                  "mtt": "100000", "mpp": "100000", "mrr": "100000",
                  "mrt": "100000", "mrp": "100000", "mtp": "100000"}

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
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    # Various sources.
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "100000", "fp": "100000"}

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
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            **dict((key[0] + "_" + key[1:], float(value))
                   for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0)

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["sourcedepthinm"] = "5.0"
        params["origintime"] = str(time)
        params["receiverdepthinm"] = "55.0"
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=5.0, origin_time=time,
            **dict((key[0] + "_" + key[1:], float(value))
                   for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
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
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=0.0,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0)

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["sourcedepthinm"] = "5.0"
        params["origintime"] = str(time)
        params["receiverdepthinm"] = "55.0"
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        request = client.fetch(_assemble_url_raw(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=5.0, origin_time=time,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
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
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=0.0,
                **dict(("_".join(key), float(value))
                       for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=0.0)

            # Moment tensor source with a couple more parameters.
            p.reset_mock()

            params["sourcedepthinm"] = "5.0"
            params["origintime"] = str(time)
            params["receiverdepthinm"] = "55.0"
            params["networkcode"] = "BW"
            params["stationcode"] = "ALTM"

            request = client.fetch(_assemble_url_raw(**params))
            assert request.code == 200

            assert p.call_count == 1
            assert p.call_args[1]["components"] == ["Z", "N", "E"]
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=5.0, origin_time=time,
                **dict(("_".join(key), float(value))
                       for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=55.0, network="BW", station="ALTM")


def test_seismograms_error_handling(all_clients):
    """
    Tests error handling of the /seismograms route. Potentially outwards
    facing thus tested rather well.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcelatitude"]
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'sourcelatitude' not given."

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["sourcelatitude"] = "A"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "sourcelatitude" in request.reason

    # No source.
    request = client.fetch(_assemble_url(**basic_parameters))
    assert request.code == 400
    assert request.reason == "No/insufficient source parameters specified"

    # Invalid receiver.
    params = copy.deepcopy(basic_parameters)
    params["receiverlatitude"] = "100"
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct receiver with " in request.reason.lower()

    # Invalid MT source.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["strike"] = "45"
    params["dip"] = "45"
    params["rake"] = "45"
    params["M0"] = "450000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct the source" in request.reason.lower()
    assert "strike/dip/rake" in request.reason.lower()

    # Invalid force source. It only works in displ_only mode but here it
    # fails earlier.
    params = copy.deepcopy(basic_parameters)
    params["fr"] = "100000"
    params["ft"] = "100000"
    params["fp"] = "100000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct force source" in request.reason.lower()

    # Could not extract seismogram.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["components"] = "ABC"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()

    # Wrong type of seismogram requested.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["unit"] = "fun"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "unit must be one of" in request.reason.lower()

    # dt is too small - protects the server from having to serve humongous
    # files.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["dt"] = "0.009"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "the smallest possible dt is 0.01" in request.reason.lower()

    # lanzcos window is too wide or too narrow.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["alanczos"] = "1"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "`alanczos` must not be smaller" in request.reason.lower()
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["alanczos"] = "21"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "`alanczos` must not be smaller" in request.reason.lower()

    # too many components raise to avoid abuse.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["components"] = "NNEERRTTZZ"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "a maximum of 5 components can be request" in request.reason.lower()

    # At least one components must be requested.
    params = copy.deepcopy(basic_parameters)
    params["mtt"] = "100000"
    params["mpp"] = "100000"
    params["mrr"] = "100000"
    params["mrt"] = "100000"
    params["mrp"] = "100000"
    params["mtp"] = "100000"
    params["components"] = ""
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "a request with no components will not re" in request.reason.lower()


def test_conversion_to_boolean_parameters(all_clients):
    """
    Boolean values can be specified in a number of ways. Test that these are
    working as expected.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "mtt": "100000",
        "mpp": "100000",
        "mrr": "100000",
        "mrt": "100000",
        "mrp": "100000",
        "mtp": "100000"}

    truth_values = ["1", "true", "TRUE", "True", "T", "t", "y", "Y"]
    false_values = ["0", "false", "FALSE", "False", "F", "f", "n", "N"]
    invalid_values = ["A", "HMMMM", "234"]

    for value in truth_values:
        params = copy.deepcopy(basic_parameters)
        params["removesourceshift"] = value
        _st = obspy.read()

        with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                        ".get_seismograms") as p:
            p.return_value = _st
            client.fetch(_assemble_url(**params))
            assert p.call_count == 1
            assert p.call_args[1]["remove_source_shift"] is True

    for value in false_values:
        params = copy.deepcopy(basic_parameters)
        params["removesourceshift"] = value
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
        params["removesourceshift"] = value
        request = client.fetch(_assemble_url(**params))
        assert request.code == 400
        assert ("parameter 'removesourceshift' could not be converted to "
                "'bool'" in request.reason.lower())


def test_object_creation_for_seismogram_route(all_clients):
    """
    Tests that the correct objects are created for the seismogram route.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    # Various sources.
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "100000", "fp": "100000"}

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
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            **dict((key[0] + "_" + key[1:], float(value))
                   for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0)
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["sourcedepthinm"] = "5.0"
        params["origintime"] = str(time)
        params["receiverdepthinm"] = "55.0"
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        request = client.fetch(_assemble_url(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=5.0, origin_time=time,
            **dict((key[0] + "_" + key[1:], float(value))
                   for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
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
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=0.0,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0)
        assert p.call_args[1]["kind"] == "displacement"
        assert p.call_args[1]["remove_source_shift"] is True
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None
        assert p.call_args[1]["a_lanczos"] == 5

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["sourcedepthinm"] = "5.0"
        params["origintime"] = str(time)
        params["receiverdepthinm"] = "55.0"
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        request = client.fetch(_assemble_url(**params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=5.0, origin_time=time,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
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
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=0.0,
                **dict(("_".join(key), float(value))
                       for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=0.0)
            assert p.call_args[1]["kind"] == "displacement"
            assert p.call_args[1]["remove_source_shift"] is True
            assert p.call_args[1]["reconvolve_stf"] is False
            assert p.call_args[1]["return_obspy_stream"] is True
            assert p.call_args[1]["dt"] is None
            assert p.call_args[1]["a_lanczos"] == 5

            # Moment tensor source with a couple more parameters.
            p.reset_mock()

            params["sourcedepthinm"] = "5.0"
            params["origintime"] = str(time)
            params["receiverdepthinm"] = "55.0"
            params["networkcode"] = "BW"
            params["stationcode"] = "ALTM"

            request = client.fetch(_assemble_url(**params))
            assert request.code == 200

            assert p.call_count == 1
            assert p.call_args[1]["components"] == ["Z", "N", "E"]
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=5.0, origin_time=time,
                **dict(("_".join(key), float(value))
                       for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
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
        params["unit"] = "acceleration"
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
        params["unit"] = "velocity"
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
        params["unit"] = "VeLoCity"
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
        params["removesourceshift"] = "False"
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
        params["alanczos"] = "20"
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
        params["alanczos"] = "2"
        params["unit"] = "ACCELERATION"
        params["removesourceshift"] = "False"
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


def test_seismograms_retrieval(all_clients):
    """
    Tests if the seismograms requested from the server are identical to the
    on requested with the local instaseis client.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    # Various sources.
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "100000", "fp": "100000"}

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)

    components = ["Z", "N", "E"]
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               components=components)
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    params["sourcedepthinm"] = "5.0"
    params["origintime"] = str(time)
    params["receiverdepthinm"] = "55.0"
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=5.0, origin_time=time,
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=55.0, network="BW", station="ALTM")
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               components=components)
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    # From strike, dip, rake
    params = copy.deepcopy(basic_parameters)
    params.update(sdr)
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"], depth_in_m=0.0)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               components=components)
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    # Moment tensor source with a couple more parameters.
    params["sourcedepthinm"] = "5.0"
    params["origintime"] = str(time)
    params["receiverdepthinm"] = "55.0"
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"

    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=5.0, origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=55.0, network="BW", station="ALTM")
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               components=components)
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    # Force source only works for displ_only databases.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params.update(fs)
        request = client.fetch(_assemble_url(**params))
        st_server = obspy.read(request.buffer)

        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receiver = instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0)
        st_db = db.get_seismograms(source=source, receiver=receiver,
                                   components=components)
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both.
            del tr_server.stats.mseed
            del tr_server.stats._format
            del tr_db.stats.instaseis
            # Sample spacing is very similar but not equal due to floating
            # point accuracy.
            np.testing.assert_allclose(tr_server.stats.delta,
                                       tr_db.stats.delta)
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(tr_server.data, tr_db.data,
                                       atol=1E-10 * tr_server.data.ptp())

        params["sourcedepthinm"] = "5.0"
        params["origintime"] = str(time)
        params["receiverdepthinm"] = "55.0"
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        request = client.fetch(_assemble_url(**params))
        st_server = obspy.read(request.buffer)

        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=5.0, origin_time=time,
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receiver = instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=55.0, network="BW", station="ALTM")
        st_db = db.get_seismograms(source=source, receiver=receiver,
                                   components=components)
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both.
            del tr_server.stats.mseed
            del tr_server.stats._format
            del tr_db.stats.instaseis
            # Sample spacing is very similar but not equal due to floating
            # point accuracy.
            np.testing.assert_allclose(tr_server.stats.delta,
                                       tr_db.stats.delta)
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(tr_server.data, tr_db.data,
                                       atol=1E-10 * tr_server.data.ptp())

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0)

    # Now test other the other parameters.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["components"] = "RTE"
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               components=["R", "T", "E"])
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta,
                                   tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["unit"] = "acceleration"
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               kind="acceleration")
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta,
                                   tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["unit"] = "velocity"
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               kind="velocity")
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta,
                                   tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["removesourceshift"] = "False"
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               remove_source_shift=False)
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta,
                                   tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["dt"] = "0.1"
    params["alanczos"] = "20"
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               dt=0.1, a_lanczos=20)
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta,
                                   tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["dt"] = "0.1"
    params["alanczos"] = "2"
    params["unit"] = "ACCELERATION"
    params["removesourceshift"] = "False"
    request = client.fetch(_assemble_url(**params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               dt=0.1, a_lanczos=2, kind="acceleration",
                               remove_source_shift=False)
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta,
                                   tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())


def test_output_formats(all_clients):
    """
    The /seismograms route can return data either as MiniSEED or as zip
    archive containing multiple files.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    basic_parameters.update(mt)

    # First don't specify the format which should result in a miniseed file.
    params = copy.deepcopy(basic_parameters)
    request = client.fetch(_assemble_url(**params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # Specifying the miniseed format also work.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "mseed"
    request = client.fetch(_assemble_url(**params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # saczip results in a folder of multiple sac files.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = client.fetch(_assemble_url(**params))
    # ObsPy needs the filename to be able to directly unpack zip files. We
    # don't have a filename here so we unpack manually.
    sac_st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        sac_st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in sac_st:
        assert tr.stats._format == "SAC"

    # Otherwise they should be identical!
    for tr in st.traces + sac_st.traces:
        del tr.stats._format
        try:
            del tr.stats.sac
        except KeyError:
            pass
        try:
            del tr.stats.mseed
        except KeyError:
            pass

    st.sort()
    sac_st.sort()

    for tr, sac_tr in zip(st, sac_st):
        # Make sure the sampling rate is approximately equal.
        np.testing.assert_allclose(tr.stats.delta, sac_tr.stats.delta)
        # Now set one to the other to make sure the following comparision is
        # meaningful
        tr.stats.delta = sac_tr.stats.delta

    # Now make sure the result is the same independent of the output format.
    assert st == sac_st

    # Once more with a couple more parameters.
    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10}
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000",
          "components": "RT", "unit": "velocity", "dt": 2, "alanczos": 3,
          "networkcode": "BW", "stationcode": "FURT"}
    basic_parameters.update(mt)

    # First don't specify the format which should result in a miniseed file.
    params = copy.deepcopy(basic_parameters)
    request = client.fetch(_assemble_url(**params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # Specifying the miniseed format also work.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "mseed"
    request = client.fetch(_assemble_url(**params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # saczip results in a folder of multiple sac files.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = client.fetch(_assemble_url(**params))
    # ObsPy needs the filename to be able to directly unpack zip files. We
    # don't have a filename here so we unpack manually.
    sac_st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        sac_st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in sac_st:
        assert tr.stats._format == "SAC"

    # Otherwise they should be identical!
    for tr in st.traces + sac_st.traces:
        del tr.stats._format
        try:
            del tr.stats.sac
        except KeyError:
            pass
        try:
            del tr.stats.mseed
        except KeyError:
            pass

    st.sort()
    sac_st.sort()

    for tr, sac_tr in zip(st, sac_st):
        # Make sure the sampling rate is approximately equal.
        np.testing.assert_allclose(tr.stats.delta, sac_tr.stats.delta)
        # Now set one to the other to make sure the following comparision is
        # meaningful
        tr.stats.delta = sac_tr.stats.delta

    # Now make sure the result is the same independent of the output format.
    assert st == sac_st


def test_coordinates_route_with_no_coordinate_callback(all_clients):
    """
    If no coordinate callback has been set, the coordinate route should
    return 404.
    """
    client = all_clients
    request = client.fetch("/coordinates?network=BW&station=FURT")
    assert request.code == 404
    assert request.reason == 'Server does not support station coordinates.'

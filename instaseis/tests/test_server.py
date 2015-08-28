#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Instaseis server.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
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

from instaseis.helpers import geocentric_to_elliptic_latitude

# Conditionally import mock either from the stdlib or as a separate library.
import sys
if sys.version_info[0] == 2:
    import mock
else:
    import unittest.mock as mock


def _assemble_url(route, **kwargs):
    """
    Helper function.
    """
    url = "/%s?" % route
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


def test_greens_function_error_handling(all_clients):
    """
    Tests error handling of the /greens_function route. Very basic for now
    """
    client = all_clients

    basic_parameters = {
        "sourcedepthinmeters": client.source_depth,
        "sourcedistanceindegrees": 20}

    # get_greens_function() only works with reciprocal DBs. So make sure we
    # get the error, but then do the other tests only for reciprocal DBs
    if not client.is_reciprocal:
        params = copy.deepcopy(basic_parameters)
        request = client.fetch(_assemble_url('greens_function', **params))
        assert request.code == 400
        assert "the database is not reciprocal" in request.reason.lower()
        return

    # Remove the sourcedistanceindegrees, required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcedistanceindegrees"]
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'sourcedistanceindegrees' not given."

    # Remove the sourcedepthinmeters, required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcedepthinmeters"]
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'sourcedepthinmeters' not given."

    # Negative source distance
    request = client.fetch(_assemble_url(
        "greens_function",
        sourcedepthinmeters=20, sourcedistanceindegrees=-30))
    assert request.code == 400
    assert request.reason == "Epicentral distance should be in [0, 180]."

    # Too far source distances.
    request = client.fetch(_assemble_url(
        "greens_function",
        sourcedepthinmeters=20, sourcedistanceindegrees=200))
    assert request.code == 400
    assert request.reason == "Epicentral distance should be in [0, 180]."

    # Negative source depth.
    request = client.fetch(_assemble_url(
        "greens_function",
        sourcedepthinmeters=-20, sourcedistanceindegrees=20))
    assert request.code == 400
    assert request.reason == "Source depth should be in [0.0, 371000.0]."

    # Too large source depth.
    request = client.fetch(_assemble_url(
        "greens_function",
        sourcedepthinmeters=2E6, sourcedistanceindegrees=20))
    assert request.code == 400
    assert request.reason == "Source depth should be in [0.0, 371000.0]."


def test_greens_function_retrieval(all_clients):
    """
    Tests if the greens functions requested from the server are identical to
    the one requested with the local instaseis client.
    """
    client = all_clients

    db = instaseis.open_db(client.filepath)

    # get_greens_function() only works with reciprocal DBs.
    if not client.is_reciprocal:
        return

    basic_parameters = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "saczip"}

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # default parameters
    params = copy.deepcopy(basic_parameters)
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 200
    # ObsPy needs the filename to be able to directly unpack zip files. We
    # don't have a filename here so we unpack manually.
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st_server:
        assert tr.stats._format == "SAC"

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params['sourcedistanceindegrees'],
        source_depth_in_m=params['sourcedepthinmeters'], origin_time=time,
        definition="seiscomp")

    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.sac
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

    # miniseed
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    for tr in st_server:
        assert tr.stats._format == "MSEED"

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params['sourcedistanceindegrees'],
        source_depth_in_m=params['sourcedepthinmeters'], origin_time=time,
        definition="seiscomp")

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

    # One with a label.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    params["label"] = "random_things"
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 200

    cd = request.headers["Content-Disposition"]
    assert cd.startswith("attachment; filename=random_things_")
    assert cd.endswith(".mseed")

    # One more with resampling parameters and different units.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    params["dt"] = 0.1
    params["kernelwidth"] = 2
    params["units"] = "acceleration"
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params['sourcedistanceindegrees'],
        source_depth_in_m=params['sourcedepthinmeters'], origin_time=time,
        definition="seiscomp", dt=0.1, kernelwidth=2, kind="acceleration")

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

    # One simulating a crash in the underlying function.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_greens_function") as p:
        def raise_err():
            raise ValueError("random crash")

        p.side_effect = raise_err
        request = client.fetch(_assemble_url('greens_function', **params))

    assert request.code == 400
    assert request.reason == ("Could not extract Green's function. Make "
                              "sure, the parameters are valid, and the depth "
                              "settings are correct.")

    # Two more simulating logic erros that should not be able to happen.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_greens_function") as p:
        st = obspy.read()
        for tr in st:
            tr.stats.starttime = obspy.UTCDateTime(1E5)

        p.return_value = st
        request = client.fetch(_assemble_url('greens_function', **params))

    assert request.code == 500
    assert request.reason == ("Starttime more than one hour before the "
                              "starttime of the seismograms.")

    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_greens_function") as p:
        st = obspy.read()
        for tr in st:
            tr.stats.starttime = obspy.UTCDateTime(0)
            tr.stats.delta = 0.0001

        p.return_value = st
        request = client.fetch(_assemble_url('greens_function', **params))

    assert request.code == 500
    assert request.reason.startswith("Endtime larger then the extracted "
                                     "endtime")


def test_phase_relative_offsets_but_no_ttimes_callback_greens_function(
        all_clients):
    client = all_clients

    # get_greens_function() only works with reciprocal DBs.
    if not client.is_reciprocal:
        return

    params = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "miniseed"}

    # Test for starttime.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations.")

    # Test for endtime.
    p = copy.deepcopy(params)
    p["endtime"] = "P%2D10"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations.")

    # Test for both.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    p["endtime"] = "S%2B10"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations.")


def test_phase_relative_offset_failures_greens_function(
        all_clients_ttimes_callback):
    """
    Tests some common failures for the phase relative offsets with the
    greens function route.
    """
    client = all_clients_ttimes_callback

    # get_greens_function() only works with reciprocal DBs.
    if not client.is_reciprocal:
        return

    params = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "miniseed"}

    # Illegal phase.
    p = copy.deepcopy(params)
    p["starttime"] = "bogus%2D10"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 400
    assert request.reason == "Invalid phase name: bogus"

    # Phase not available at that distance.
    p = copy.deepcopy(params)
    p["starttime"] = "Pdiff%2D10"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 400
    assert request.reason == (
        "No Green's function extracted for the given phase relative offsets. "
        "This could either be due to the chosen phase not existing for the "
        "specific source-receiver geometry or arriving too late/with too "
        "large offsets if the database is not long enough.")


def test_phase_relative_offsets_greens_function(all_clients_ttimes_callback):
    """
    Test phase relative offsets with the green's function route.

    + must be encoded with %2B
    - must be encoded with %2D
    """
    client = all_clients_ttimes_callback

    # Only for reciprocal databases.
    if not client.is_reciprocal:
        return

    # At a distance of 50 degrees and with a source depth of 300 km:
    # P: 504.357 seconds
    # PP: 622.559 seconds
    # sPKiKP: 1090.081 seconds
    params = {
        "sourcedepthinmeters": 300000,
        "sourcedistanceindegrees": 50,
        "format": "miniseed", "dt": 0.1}

    # Normal seismogram.
    p = copy.deepcopy(params)
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    starttime, endtime = tr.stats.starttime, tr.stats.endtime

    # Start 10 seconds before the P arrival.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Starts 10 seconds after the P arrival
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Ends 15 seconds before the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2D15"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 15)) < 0.1

    # Ends 15 seconds after the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2B15"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1

    # Starts 5 seconds before the PP and ends 2 seconds after the sPKiKP phase.
    p = copy.deepcopy(params)
    p["starttime"] = "PP%2D5"
    p["endtime"] = "sPKiKP%2B2"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 622.559 - 5)) < 0.1
    assert abs((tr.stats.endtime) - (starttime + 1090.081 + 2)) < 0.1

    # Combinations with relative end times are also possible. Relative start
    # times are always relative to the origin time so it does not matter in
    # that case.
    p = copy.deepcopy(params)
    p["starttime"] = "PP%2D5"
    p["endtime"] = 10.0
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 622.559 - 5)) < 0.1
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 5 + 10)) < 0.1

    # Nonetheless, also test the other combination of relative start time
    # and phase relative endtime.
    p = copy.deepcopy(params)
    p["starttime"] = "10"
    p["endtime"] = "PP%2B15"
    request = client.fetch(_assemble_url('greens_function', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime + 10
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1


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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'sourcelatitude' not given."

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["sourcelatitude"] = "A"
    request = client.fetch(_assemble_url('seismograms_raw', **params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "sourcelatitude" in request.reason

    # No source.
    request = client.fetch(_assemble_url('seismograms_raw',
                           **basic_parameters))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["strike"] = "45"
    params["dip"] = "45"
    params["rake"] = "45"
    params["M0"] = "450000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
        request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    mt = {"mtt": "100000", "mpp": "200000", "mrr": "300000",
          "mrt": "400000", "mrp": "500000", "mtp": "600000"}
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
        request = client.fetch(_assemble_url('seismograms_raw', **params))
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
        request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
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
    request = client.fetch(_assemble_url('seismograms_raw', **parameters))
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
    mt = {"mtt": "100000", "mpp": "200000", "mrr": "300000",
          "mrt": "400000", "mrp": "500000", "mtp": "600000"}
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    with mock.patch("instaseis.instaseis_db.InstaseisDB._get_seismograms") \
            as p:
        _st = obspy.read()
        for tr in _st:
            tr.stats.starttime = obspy.UTCDateTime(0)
        data = {}
        data["mu"] = 1.0
        for tr in _st:
            data[tr.stats.channel[-1]] = tr.data
        p.return_value = data

        # Moment tensor source.
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        request = client.fetch(_assemble_url('seismograms_raw', **params))
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

        request = client.fetch(_assemble_url('seismograms_raw', **params))
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
        request = client.fetch(_assemble_url('seismograms_raw', **params))
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

        request = client.fetch(_assemble_url('seismograms_raw', **params))
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
            request = client.fetch(_assemble_url('seismograms_raw', **params))
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

            request = client.fetch(_assemble_url('seismograms_raw', **params))
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
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcelatitude"]
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == \
        "The following required parameters are missing: 'sourcelatitude'"

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["sourcelatitude"] = "A"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "sourcelatitude" in request.reason

    # No source.
    request = client.fetch(_assemble_url('seismograms', **basic_parameters))
    assert request.code == 400
    assert request.reason == (
        "One of the following has to be given: 'eventid', "
        "'sourcedoublecouple', 'sourceforce', 'sourcemomenttensor'")

    # Invalid receiver.
    params = copy.deepcopy(basic_parameters)
    params["receiverlatitude"] = "100"
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "could not construct receiver with " in request.reason.lower()

    # Invalid MT source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = "45,45,45,450000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "could not construct the source" in request.reason.lower()
    assert "strike/dip/rake" in request.reason.lower()

    # Invalid force source. It only works in displ_only mode but here it
    # fails earlier.
    params = copy.deepcopy(basic_parameters)
    params["sourceforce"] = "100000,100000,100000"
    params["sourcelatitude"] = "100"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "could not construct force source" in request.reason.lower()

    # Could not extract seismogram.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["components"] = "ABC"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()

    # Wrong type of seismogram requested.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["units"] = "fun"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "unit must be one of" in request.reason.lower()

    # dt is too small - protects the server from having to serve humongous
    # files.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["dt"] = "0.009"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "the smallest possible dt is 0.01" in request.reason.lower()

    # interpolation kernel width is too wide or too narrow.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["kernelwidth"] = "0"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "`kernelwidth` must not be smaller" in request.reason.lower()
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["kernelwidth"] = "21"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "`kernelwidth` must not be smaller" in request.reason.lower()

    # too many components raise to avoid abuse.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["components"] = "NNEERRTTZZ"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "a maximum of 5 components can be request" in request.reason.lower()

    # At least one components must be requested.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["components"] = ""
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "a request with no components will not re" in request.reason.lower()


def test_object_creation_for_seismogram_route(all_clients):
    """
    Tests that the correct objects are created for the seismogram route.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10}

    dt = 24.724845445855724

    # Various sources.
    mt = {"mrr": "100000", "mtt": "200000", "mpp": "300000",
          "mrt": "400000", "mrp": "500000", "mtp": "600000"}
    mt_param = "100000,200000,300000,400000,500000,600000"
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    sdr_param = "10,20,30,1000000"
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}
    fs_param = "100000,200000,300000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    with mock.patch("instaseis.instaseis_db.InstaseisDB.get_seismograms") \
            as p:
        _st = obspy.read()
        for tr in _st:
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt
        p.return_value = _st

        # Moment tensor source.
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=client.source_depth,
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict((key[0] + "_" + key[1:], float(value))
                   for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0, network="XX", station="SYN")
        assert p.call_args[1]["kind"] == "displacement"
        # Remove source shift is always False.
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        # Source depth can only be set for reciprocal databases.
        if client.is_reciprocal:
            params["sourcedepthinmeters"] = "5.0"
            params["receiverdepthinmeters"] = "0.0"
        # Receiverdepth setting only valid for forward databases.
        else:
            params["receiverdepthinmeters"] = "55.0"

        params["origintime"] = str(time)
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        # We need to adjust the time values for the mock here.
        _st.traces = obspy.read().traces
        for tr in _st:
            tr.stats.starttime = time - 1 - 7 * dt
            tr.stats.delta = dt

        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=params["sourcedepthinmeters"],
            origin_time=time,
            **dict((key[0] + "_" + key[1:], float(value))
                   for (key, value) in mt.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=params["receiverdepthinmeters"],
            network="BW", station="ALTM")
        assert p.call_args[1]["kind"] == "displacement"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        # From strike, dip, rake
        p.reset_mock()
        _st.traces = obspy.read().traces
        for tr in _st:
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt

        params = copy.deepcopy(basic_parameters)
        params["sourcedoublecouple"] = sdr_param
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=basic_parameters["sourcedepthinmeters"],
                origin_time=obspy.UTCDateTime(1900, 1, 1),
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0, network="XX", station="SYN")
        assert p.call_args[1]["kind"] == "displacement"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        # Moment tensor source with a couple more parameters.
        p.reset_mock()
        _st.traces = obspy.read().traces
        for tr in _st:
            tr.stats.starttime = time - 1 - 7 * dt
            tr.stats.delta = dt

        # Source depth can only be set for reciprocal databases.
        if client.is_reciprocal:
            params["sourcedepthinmeters"] = "5.0"
            params["receiverdepthinmeters"] = "0.0"
        else:
            params["receiverdepthinmeters"] = "55.0"

        params["origintime"] = str(time)
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=params["sourcedepthinmeters"],
                origin_time=time,
                **dict((key, float(value)) for (key, value) in sdr.items()))
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=params["receiverdepthinmeters"],
            network="BW", station="ALTM")
        assert p.call_args[1]["kind"] == "displacement"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        # If the seismic moment is not given, it will be set to 1E19
        p.reset_mock()
        _st.traces = obspy.read().traces
        for tr in _st:
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt

        params = copy.deepcopy(basic_parameters)
        params["sourcedoublecouple"] = "10,10,10"
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200

        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["source"] == \
            instaseis.Source.from_strike_dip_rake(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=basic_parameters["sourcedepthinmeters"],
                origin_time=obspy.UTCDateTime(1900, 1, 1),
                strike=10, dip=10, rake=10, M0=1E19)
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0, network="XX", station="SYN")
        assert p.call_args[1]["kind"] == "displacement"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        # Force source only works for displ_only databases.
        if "displ_only" in client.filepath:
            p.reset_mock()
            _st.traces = obspy.read().traces
            for tr in _st:
                tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
                tr.stats.delta = dt

            params = copy.deepcopy(basic_parameters)
            params["sourceforce"] = fs_param
            request = client.fetch(_assemble_url('seismograms', **params))
            assert request.code == 200

            assert p.call_count == 1
            assert p.call_args[1]["components"] == ["Z", "N", "E"]
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
                **dict(("_".join(key), float(value))
                       for (key, value) in fs.items()))
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=0.0, network="XX", station="SYN")
            assert p.call_args[1]["kind"] == "displacement"
            # Remove source shift is always False
            assert p.call_args[1]["remove_source_shift"] is False
            assert p.call_args[1]["reconvolve_stf"] is False
            assert p.call_args[1]["return_obspy_stream"] is True
            assert p.call_args[1]["dt"] is None

            # Moment tensor source with a couple more parameters.
            p.reset_mock()
            _st.traces = obspy.read().traces
            for tr in _st:
                tr.stats.starttime = time - 1 - 7 * dt
                tr.stats.delta = dt

            params["sourcedepthinmeters"] = "5.0"
            params["origintime"] = str(time)
            params["receiverdepthinmeters"] = "0.0"
            params["networkcode"] = "BW"
            params["stationcode"] = "ALTM"

            request = client.fetch(_assemble_url('seismograms', **params))
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
                depth_in_m=0.0, network="BW", station="ALTM")
            assert p.call_args[1]["kind"] == "displacement"
            # Remove source shift is always False
            assert p.call_args[1]["remove_source_shift"] is False
            assert p.call_args[1]["reconvolve_stf"] is False
            assert p.call_args[1]["return_obspy_stream"] is True
            assert p.call_args[1]["dt"] is None

        # Now test other the other parameters.
        p.reset_mock()

        _st.traces = obspy.read().traces
        for tr in _st:
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt

        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["components"] = "RTE"
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["R", "T", "E"]
        assert p.call_args[1]["kind"] == "displacement"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["units"] = "acceleration"
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "acceleration"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["units"] = "velocity"
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "velocity"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["units"] = "VeLoCity"
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "velocity"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] is None

        p.reset_mock()
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["dt"] = "0.1"
        params["kernelwidth"] = "20"
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "displacement"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] == 0.1
        assert p.call_args[1]["kernelwidth"] == 20

        p.reset_mock()
        _st.traces = obspy.read().traces
        for tr in _st:
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["dt"] = "0.1"
        params["kernelwidth"] = "2"
        params["units"] = "ACCELERATION"
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert p.call_count == 1
        assert p.call_args[1]["components"] == ["Z", "N", "E"]
        assert p.call_args[1]["kind"] == "acceleration"
        # Remove source shift is always False
        assert p.call_args[1]["remove_source_shift"] is False
        assert p.call_args[1]["reconvolve_stf"] is False
        assert p.call_args[1]["return_obspy_stream"] is True
        assert p.call_args[1]["dt"] == 0.1
        assert p.call_args[1]["kernelwidth"] == 2


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
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "format": "miniseed"}

    # Various sources.
    mt = {"mrr": "100000", "mtt": "200000", "mpp": "300000",
          "mrt": "400000", "mrp": "500000", "mtp": "600000"}
    mt_param = "100000,200000,300000,400000,500000,600000"
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    sdr_param = "10,20,30,1000000"
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}
    fs_param = "100000,200000,300000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    request = client.fetch(_assemble_url('seismograms', **params))
    st_server = obspy.read(request.buffer)

    components = ["Z", "N", "E"]
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0, network="XX", station="SYN")
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

    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
        params["receiverdepthinmeters"] = "0.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)
        params["receiverdepthinmeters"] = "55.0"

    params["origintime"] = str(time)
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"
    request = client.fetch(_assemble_url('seismograms', **params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=params["receiverdepthinmeters"],
        network="BW", station="ALTM")
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
    params["sourcedoublecouple"] = sdr_param
    request = client.fetch(_assemble_url('seismograms', **params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"], depth_in_m=0.0,
        network="XX", station="SYN")
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
    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
        params["receiverdepthinmeters"] = "0.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)
        params["receiverdepthinmeters"] = "55.0"

    params["origintime"] = str(time)
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"

    request = client.fetch(_assemble_url('seismograms', **params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=params["receiverdepthinmeters"],
        network="BW", station="ALTM")
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
        params["sourceforce"] = fs_param
        request = client.fetch(_assemble_url('seismograms', **params))
        st_server = obspy.read(request.buffer)

        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receiver = instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0, network="XX", station="SYN")
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

        if client.is_reciprocal:
            params["sourcedepthinmeters"] = "5.0"
            params["receiverdepthinmeters"] = "0.0"
        else:
            params["sourcedepthinmeters"] = "0.0"
            params["receiverdepthinmeters"] = "55.0"

        params["origintime"] = str(time)
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"

        request = client.fetch(_assemble_url('seismograms', **params))
        st_server = obspy.read(request.buffer)

        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=params["sourcedepthinmeters"],
            origin_time=time,
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receiver = instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=params["receiverdepthinmeters"],
            network="BW", station="ALTM")
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
        depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0, network="XX", station="SYN")

    # Now test other the other parameters.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["components"] = "RTE"
    request = client.fetch(_assemble_url('seismograms', **params))
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
    params["sourcemomenttensor"] = mt_param
    params["units"] = "acceleration"
    request = client.fetch(_assemble_url('seismograms', **params))
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
    params["sourcemomenttensor"] = mt_param
    params["units"] = "velocity"
    request = client.fetch(_assemble_url('seismograms', **params))
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
    params["sourcemomenttensor"] = mt_param
    params["dt"] = "0.1"
    params["kernelwidth"] = "1"
    request = client.fetch(_assemble_url('seismograms', **params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               dt=0.1, kernelwidth=1)
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
    params["sourcemomenttensor"] = mt_param
    params["dt"] = "0.1"
    params["kernelwidth"] = "2"
    params["units"] = "ACCELERATION"
    request = client.fetch(_assemble_url('seismograms', **params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(source=source, receiver=receiver,
                               dt=0.1, kernelwidth=2, kind="acceleration",
                               remove_source_shift=True)
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
    archive containing multiple SAC files.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000"}

    # First try to get a MiniSEED file.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    request = client.fetch(_assemble_url('seismograms', **params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # saczip results in a folder of multiple sac files.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = client.fetch(_assemble_url('seismograms', **params))
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
        # Now set one to the other to make sure the following comparison is
        # meaningful
        tr.stats.delta = sac_tr.stats.delta

    # Now make sure the result is the same independent of the output format.
    assert st == sac_st

    # Specifying the saczip format also work.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = client.fetch(_assemble_url('seismograms', **params))
    sac_st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        sac_st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in sac_st:
        assert tr.stats._format == "SAC"

    # Once more with a couple more parameters.
    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10}
    mt = {"sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
          "components": "RT", "units": "velocity", "dt": 2, "kernelwidth": 3,
          "networkcode": "BW", "stationcode": "FURT"}
    basic_parameters.update(mt)

    # First get a MiniSEED file.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    request = client.fetch(_assemble_url('seismograms', **params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # saczip results in a folder of multiple sac files.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = client.fetch(_assemble_url('seismograms', **params))
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

    # Specifying the saczip format also work.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = client.fetch(_assemble_url('seismograms', **params))
    sac_st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        sac_st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in sac_st:
        assert tr.stats._format == "SAC"


def test_coordinates_route_with_no_coordinate_callback(all_clients):
    """
    If no coordinate callback has been set, the coordinate route should
    return 404.
    """
    client = all_clients
    request = client.fetch("/coordinates?network=BW&station=FURT")
    assert request.code == 404
    assert request.reason == 'Server does not support station coordinates.'


def test_coordinates_route_with_stations_coordinates_callback(
        all_clients_station_coordinates_callback):
    """
    Tests the /coordinates route.
    """
    client = all_clients_station_coordinates_callback

    # 404 is returned if no coordinates are found.
    request = client.fetch("/coordinates?network=BW&station=FURT")
    assert request.code == 404
    assert request.reason == 'No coordinates found satisfying the query.'

    # Single station.
    request = client.fetch("/coordinates?network=IU&station=ANMO")
    assert request.code == 200
    # Assert the GeoJSON content-type.
    assert request.headers["Content-Type"] == "application/vnd.geo+json"
    stations = json.loads(str(request.body.decode("utf8")))

    assert stations == {
        'features': [
            {'geometry': {'coordinates': [-106.4572, 34.94591],
                          'type': 'Point'},
             'properties': {'network_code': 'IU', 'station_code': 'ANMO'},
             'type': 'Feature'}
        ],
        'type': 'FeatureCollection'}

    # Multiple stations with wildcard searches.
    request = client.fetch("/coordinates?network=IU,B*&station=ANT*,ANM?")
    assert request.code == 200
    # Assert the GeoJSON content-type.
    assert request.headers["Content-Type"] == "application/vnd.geo+json"
    stations = json.loads(str(request.body.decode("utf8")))

    assert stations == {
        'features': [
            {'geometry': {'coordinates': [32.7934, 39.868], 'type': 'Point'},
             'properties': {'network_code': 'IU', 'station_code': 'ANTO'},
             'type': 'Feature'},
            {'geometry': {'coordinates': [-106.4572, 34.94591],
                          'type': 'Point'},
             'properties': {'network_code': 'IU', 'station_code': 'ANMO'},
             'type': 'Feature'}],
        'type': 'FeatureCollection'}


def test_cors_headers(all_clients_all_callbacks):
    """
    Check that all routes return CORS headers.
    """
    client = all_clients_all_callbacks

    request = client.fetch("/")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = client.fetch("/info")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = client.fetch("/coordinates?network=IU&station=ANMO")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = client.fetch("/event?id=B071791B")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = client.fetch(
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=%i&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phase=P" % client.source_depth)
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    # raw seismograms route
    params = {
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
    request = client.fetch(_assemble_url('seismograms_raw', **params))
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    # standard seismograms route
    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"


def test_cors_headers_failing_requests(all_clients_all_callbacks):
    """
    Check that all routes return CORS headers also for failing requests.
    """
    client = all_clients_all_callbacks

    request = client.fetch("/coordinates")
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = client.fetch("/event")
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = client.fetch("/ttimes")
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    # raw seismograms route
    params = {}
    request = client.fetch(_assemble_url('seismograms_raw', **params))
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    # standard seismograms route
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"


def test_multiple_seismograms_retrieval_no_format_given(
        all_clients_station_coordinates_callback):
    """
    Tests  the retrieval of multiple station in one request with no passed
    format parameter. This results in saczip return values.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {"sourcelatitude": 10, "sourcelongitude": 10,
                        "sourcedepthinmeters": client.source_depth}

    # Various sources.
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    mt_param = "100000,100000,100000,100000,100000,100000"
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    sdr_param = "10,10,10,1000000"
    fs = {"fr": "100000", "ft": "100000", "fp": "100000"}
    fs_param = "100000,100000,100000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"

    # Default format is MiniSEED>
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    st_server.sort()

    components = ["Z", "N", "E"]
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receivers = [
        instaseis.Receiver(latitude=39.868, longitude=32.7934,
                           depth_in_m=0.0, network="IU", station="ANTO"),
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have 6 Stream objects.
    assert len(st_db) == 6
    assert len(st_server) == 6
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.sac
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

    # Strike/dip/rake source, "RT" components
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = sdr_param
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"
    params["components"] = "RT"
    # A couple more parameters.
    if client.is_reciprocal is True:
        params["sourcedepthinmeters"] = "5.0"
    params["origintime"] = str(time)

    # Default format is MiniSEED>
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    st_server.sort()

    components = ["R", "T"]
    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=5.0, origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receivers = [
        instaseis.Receiver(latitude=39.868, longitude=32.7934,
                           depth_in_m=0.0, network="IU", station="ANTO"),
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have only 4 Stream objects.
    assert len(st_db) == 4
    assert len(st_server) == 4
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both. This also assures both have
        # the  miniseed format.
        del tr_server.stats.sac
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
    # Force source, all 5 components.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params["sourceforce"] = fs_param
        # This will return two stations.
        params["network"] = "IU,B*"
        params["station"] = "ANT*,ANM?"
        params["components"] = "NEZRT"

        # Default format is MiniSEED>
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert request.headers["Content-Type"] == "application/zip"
        st_server = obspy.Stream()
        zip_obj = zipfile.ZipFile(request.buffer)
        for name in zip_obj.namelist():
            st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
        st_server.sort()

        components = ["N", "E", "Z", "R", "T"]
        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receivers = [
            instaseis.Receiver(latitude=39.868, longitude=32.7934,
                               depth_in_m=0.0, network="IU", station="ANTO"),
            instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                               depth_in_m=0.0, network="IU", station="ANMO")]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(source=source, receiver=receiver,
                                        components=components)
        st_db.sort()

        # Should now have 10 Stream objects.
        assert len(st_db) == 10
        assert len(st_server) == 10
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both.
            del tr_server.stats.sac
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


def test_multiple_seismograms_retrieval_no_format_given_single_station(
        all_clients_station_coordinates_callback):
    """
    Tests  the retrieval of multiple station in one request with no passed
    format parameter. This results in sac return values.

    In this case the query is constructed so that it only returns a single
    station.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {"sourcelatitude": 10, "sourcelongitude": 10,
                        "sourcedepthinmeters": client.source_depth}

    # Various sources.
    mt = {"mrr": "100000", "mtt": "200000", "mpp": "300000",
          "mrt": "400000", "mrp": "500000", "mtp": "600000"}
    mt_param = "100000,200000,300000,400000,500000,600000"
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    sdr_param = "10,20,30,1000000"
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}
    fs_param = "100000,200000,300000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    # This will return two stations.
    params["network"] = "IU"
    params["station"] = "ANMO"

    # Default format is saczip.
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    st_server.sort()

    components = ["Z", "N", "E"]
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receivers = [
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have 3 Stream objects.
    assert len(st_db) == 3
    assert len(st_server) == 3
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.sac
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

    # Strike/dip/rake source, "RT" components
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = sdr_param
    # This will return two stations.
    params["network"] = "IU"
    params["station"] = "ANMO"
    params["components"] = "RT"
    # A couple more parameters.
    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)
    params["origintime"] = str(time)

    # Default format is MiniSEED>
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    st_server.sort()

    components = ["R", "T"]
    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receivers = [
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have only 2 Stream objects.
    assert len(st_db) == 2
    assert len(st_server) == 2
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.sac
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
    # Force source, all 5 components.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params["sourceforce"] = fs_param
        # This will return two stations.
        params["network"] = "IU"
        params["station"] = "ANMO"
        params["components"] = "NEZRT"

        # Default format is MiniSEED>
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert request.headers["Content-Type"] == "application/zip"
        st_server = obspy.Stream()
        zip_obj = zipfile.ZipFile(request.buffer)
        for name in zip_obj.namelist():
            st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
        st_server.sort()

        components = ["N", "E", "Z", "R", "T"]
        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receivers = [
            instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                               depth_in_m=0.0, network="IU", station="ANMO")]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(source=source, receiver=receiver,
                                        components=components)
        st_db.sort()

        # Should now have 5 Stream objects.
        assert len(st_db) == 5
        assert len(st_server) == 5
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both.
            del tr_server.stats.sac
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


def test_multiple_seismograms_retrieval_mseed_format(
        all_clients_station_coordinates_callback):
    """
    Tests  the retrieval of multiple station in one request with the mseed
    format parameter.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {"sourcelatitude": 10, "sourcelongitude": 10,
                        "format": "miniseed",
                        "sourcedepthinmeters": client.source_depth}

    # Various sources.
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    mt_param = "100000,100000,100000,100000,100000,100000"
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    sdr_param = "10,10,10,1000000"
    fs = {"fr": "100000", "ft": "100000", "fp": "100000"}
    fs_param = "100000,100000,100000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"

    # Default format is MiniSEED>
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/octet-stream"
    st_server = obspy.read(request.buffer)
    st_server.sort()

    components = ["Z", "N", "E"]
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receivers = [
        instaseis.Receiver(latitude=39.868, longitude=32.7934,
                           depth_in_m=0.0, network="IU", station="ANTO"),
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have 6 Stream objects.
    assert len(st_db) == 6
    assert len(st_server) == 6
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both. This also assures both have
        # the  miniseed format.
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

    # Strike/dip/rake source, "RT" components
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = sdr_param
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"
    params["components"] = "RT"
    # A couple more parameters.
    params["origintime"] = str(time)
    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)

    # Default format is MiniSEED>
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/octet-stream"
    st_server = obspy.read(request.buffer)
    st_server.sort()

    components = ["R", "T"]
    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receivers = [
        instaseis.Receiver(latitude=39.868, longitude=32.7934,
                           depth_in_m=0.0, network="IU", station="ANTO"),
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have only 4 Stream objects.
    assert len(st_db) == 4
    assert len(st_server) == 4
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both. This also assures both have
        # the  miniseed format.
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
    # Force source, all 5 components.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params["sourceforce"] = fs_param
        # This will return two stations.
        params["network"] = "IU,B*"
        params["station"] = "ANT*,ANM?"
        params["components"] = "NEZRT"

        # Default format is MiniSEED>
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert request.headers["Content-Type"] == "application/octet-stream"
        st_server = obspy.read(request.buffer)
        st_server.sort()

        components = ["N", "E", "Z", "R", "T"]
        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receivers = [
            instaseis.Receiver(latitude=39.868, longitude=32.7934,
                               depth_in_m=0.0, network="IU", station="ANTO"),
            instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                               depth_in_m=0.0, network="IU", station="ANMO")]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(source=source, receiver=receiver,
                                        components=components)
        st_db.sort()

        # Should now have 10 Stream objects.
        assert len(st_db) == 10
        assert len(st_server) == 10
        assert len(st_server) == 10
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both. This also assures both
            # have the  miniseed format.
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


def test_multiple_seismograms_retrieval_saczip_format(
        all_clients_station_coordinates_callback):
    """
    Tests  the retrieval of multiple station in one request with the saczip
    format parameter.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {"sourcelatitude": 10, "sourcelongitude": 10,
                        "sourcedepthinmeters": client.source_depth,
                        "format": "saczip"}

    # Various sources.
    mt = {"mtt": "100000", "mpp": "100000", "mrr": "100000",
          "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    mt_param = "100000,100000,100000,100000,100000,100000"
    sdr = {"strike": "10", "dip": "10", "rake": "10", "M0": "1000000"}
    sdr_param = "10,10,10,1000000"
    fs = {"fr": "100000", "ft": "100000", "fp": "100000"}
    fs_param = "100000,100000,100000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"

    # Default format is MiniSEED>
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"

    # ObsPy needs the filename to be able to directly unpack zip files. We
    # don't have a filename here so we unpack manually.
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st_server:
        assert tr.stats._format == "SAC"
    st_server.sort()

    components = ["Z", "N", "E"]
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key[0] + "_" + key[1:], float(value))
               for (key, value) in mt.items()))
    receivers = [
        instaseis.Receiver(latitude=39.868, longitude=32.7934,
                           depth_in_m=0.0, network="IU", station="ANTO"),
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have 6 Stream objects.
    assert len(st_db) == 6
    assert len(st_server) == 6
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.sac
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

    # Strike/dip/rake source, "RT" components
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = sdr_param
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"
    params["components"] = "RT"
    # A couple more parameters.
    params["origintime"] = str(time)
    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)

    # Default format is saczip.
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"

    # ObsPy needs the filename to be able to directly unpack zip files. We
    # don't have a filename here so we unpack manually.
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st_server:
        assert tr.stats._format == "SAC"
    st_server.sort()

    components = ["R", "T"]
    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=5.0, origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()))
    receivers = [
        instaseis.Receiver(latitude=39.868, longitude=32.7934,
                           depth_in_m=0.0, network="IU", station="ANTO"),
        instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                           depth_in_m=0.0, network="IU", station="ANMO")]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver,
                                    components=components)
    st_db.sort()

    # Should now have only 4 Stream objects.
    assert len(st_db) == 4
    assert len(st_server) == 4
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.sac
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
    # Force source, all 5 components.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params["sourceforce"] = fs_param
        # This will return two stations.
        params["network"] = "IU,B*"
        params["station"] = "ANT*,ANM?"
        params["components"] = "NEZRT"

        # Default format is MiniSEED>
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 200
        assert request.headers["Content-Type"] == "application/zip"

        # ObsPy needs the filename to be able to directly unpack zip files. We
        # don't have a filename here so we unpack manually.
        st_server = obspy.Stream()
        zip_obj = zipfile.ZipFile(request.buffer)
        for name in zip_obj.namelist():
            st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
        for tr in st_server:
            assert tr.stats._format == "SAC"
        st_server.sort()

        components = ["N", "E", "Z", "R", "T"]
        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0, origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(("_".join(key), float(value))
                   for (key, value) in fs.items()))
        receivers = [
            instaseis.Receiver(latitude=39.868, longitude=32.7934,
                               depth_in_m=0.0, network="IU", station="ANTO"),
            instaseis.Receiver(latitude=34.94591, longitude=-106.4572,
                               depth_in_m=0.0, network="IU", station="ANMO")]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(source=source, receiver=receiver,
                                        components=components)
        st_db.sort()

        # Should now have 10 Stream objects.
        assert len(st_db) == 10
        assert len(st_server) == 10
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both.
            del tr_server.stats.sac
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


def test_multiple_seismograms_retrieval_invalid_format(
        all_clients_station_coordinates_callback):
    """
    Tests  the retrieval of multiple station with an invalid format.
    """
    client = all_clients_station_coordinates_callback

    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "format": "bogus"}
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"

    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == "Format must either be 'miniseed' or 'saczip'."


def test_multiple_seismograms_retrieval_no_stations(
        all_clients_station_coordinates_callback):
    """
    Tests  the retrieval of multiple station where the request ends up in no
    found stations.
    """
    client = all_clients_station_coordinates_callback

    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000"}
    # This will return two stations.
    params["network"] = "HE"
    params["station"] = "LLO"

    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 404
    assert request.reason == "No coordinates found satisfying the query."


def test_unknown_parameter_raises(all_clients):
    """
    Unknown parameters should raise.
    """
    client = all_clients

    # Normal request works fine.
    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "receiverlatitude": -10,
        "receiverlongitude": -10, "mtt": "100000", "mpp": "100000",
        "mrr": "100000", "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    request = client.fetch(_assemble_url('seismograms_raw', **params))
    assert request.code == 200

    # Adding a random other parameter raises
    params["bogus"] = "bogus"
    params["random"] = "stuff"
    request = client.fetch(_assemble_url('seismograms_raw', **params))
    assert request.code == 400
    assert request.reason == ("The following unknown parameters have been "
                              "passed: 'bogus', 'random'")

    # Same with /seismograms route.
    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "receiverlatitude": -10,
        "receiverlongitude": -10, "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200

    # Adding a random other parameter raises
    params["random"] = "stuff"
    params["bogus"] = "bogus"
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == ("The following unknown parameters have been "
                              "passed: 'bogus', 'random'")


def test_passing_duplicate_parameter_raises(all_clients):
    """
    While valid with HTTP, duplicate parameters are not allowed within
    instaseis. This should thus raise an error to avoid confusion of users.
    """
    client = all_clients

    # Normal request works fine.
    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "receiverlatitude": -10,
        "receiverlongitude": -10, "mtt": "100000", "mpp": "100000",
        "mrr": "100000", "mrt": "100000", "mrp": "100000", "mtp": "100000"}
    url = _assemble_url('seismograms_raw', **params)
    request = client.fetch(url)
    assert request.code == 200

    # Adding two duplicate parameters raises.
    url += "&receiverlatitude=10&mrt=10"
    request = client.fetch(url)
    assert request.code == 400
    assert request.reason == ("Duplicate parameters: 'mrt', "
                              "'receiverlatitude'")

    # Same with /seismograms route.
    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "receiverlatitude": -10,
        "receiverlongitude": -10, "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000"}
    url = _assemble_url('seismograms', **params)
    request = client.fetch(url)
    assert request.code == 200

    # Adding two duplicate parameters raises.
    url += "&receiverlatitude=10&sourcemomenttensor=10"
    request = client.fetch(url)
    assert request.code == 400
    assert request.reason == (
        "Duplicate parameters: 'receiverlatitude', 'sourcemomenttensor'")


def test_passing_invalid_time_settings_raises(all_clients):
    """
    Tests that invalid time settings raise.
    """
    origin_time = obspy.UTCDateTime(2015, 1, 1)
    client = all_clients
    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "receiverlatitude": -10,
        "receiverlongitude": -10, "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "origintime": str(origin_time)}

    # This should work fine.
    url = _assemble_url('seismograms', **params)
    request = client.fetch(url)
    assert request.code == 200

    # The remainder should not.
    p = copy.deepcopy(params)
    p["starttime"] = str(origin_time + 1E6)
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    assert request.code == 400
    assert request.reason == ("The `starttime` must be before the seismogram "
                              "ends.")

    p = copy.deepcopy(params)
    p["endtime"] = str(origin_time - 1E6)
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    assert request.code == 400
    assert request.reason == ("The end time of the seismograms lies outside "
                              "the allowed range.")

    p = copy.deepcopy(params)
    p["starttime"] = str(origin_time - 3800)
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    assert request.code == 400
    assert request.reason == ("The seismogram can start at the maximum one "
                              "hour before the origin time.")


def test_time_settings_for_seismograms_route(all_clients):
    """
    Tests the advanced time settings.
    """
    client = all_clients

    origin_time = obspy.UTCDateTime(2015, 1, 1)

    client = all_clients
    # Resample to 1Hz to simplify the logic.
    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "receiverlatitude": -10,
        "receiverlongitude": -10, "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 1.0, "kernelwidth": 1, "origintime": str(origin_time),
        "format": "miniseed"}

    p = copy.deepcopy(params)
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    st = obspy.read(request.buffer)

    # This should start at the origin time.
    for tr in st:
        assert tr.stats.starttime == origin_time

    # Different starttime.
    p = copy.deepcopy(params)
    p["starttime"] = origin_time - 10
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time - 10

    # Can also be given as a float in which case it will be interpreted as
    # an offset to the origin time.
    p = copy.deepcopy(params)
    p["starttime"] = -10
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time - 10

    org_endtime = st[0].stats.endtime

    # Nonetheless the seismograms should still be identical for the time
    # starting at the origin time.
    for tr1, tr2 in zip(st, st_2.slice(starttime=origin_time)):
        del tr1.stats.mseed
        del tr2.stats.mseed
        del tr2.stats.processing
        assert tr1 == tr2

    # Test endtime settings.
    p = copy.deepcopy(params)
    p["endtime"] = origin_time + 10
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time
        assert tr.stats.endtime == origin_time + 10

    # Can also be done by passing a float which will be interpreted as an
    # offset in respect to the starttime.
    p = copy.deepcopy(params)
    p["endtime"] = 13.0
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time
        assert tr.stats.endtime == origin_time + 13

    # If starttime is given, the duration is relative to the starttime.
    p = copy.deepcopy(params)
    p["endtime"] = 10
    p["starttime"] = origin_time - 5
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time - 5
        assert tr.stats.endtime == origin_time + 5

    # Will be padded with zeros in the front. Attempting to pad with zeros
    # in the back will raise an error but that is tested elsewhere.
    p = copy.deepcopy(params)
    p["starttime"] = origin_time - 1800
    url = _assemble_url('seismograms', **p)
    request = client.fetch(url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time - 1800
        assert tr.stats.endtime == org_endtime
        np.testing.assert_allclose(tr.data[:100], np.zeros(100))


def test_event_route_with_no_event_callback(all_clients):
    """
    If no event information callback has been set, the event route should
    return 404.
    """
    client = all_clients
    request = client.fetch("/event?id=B071791B")
    assert request.code == 404
    assert request.reason == 'Server does not support event information.'


def test_event_route_with_event_coordinates_callback(
        all_clients_event_callback):
    """
    Tests the /event route.
    """
    client = all_clients_event_callback

    # Missing 'id' parameter.
    request = client.fetch("/event")
    assert request.code == 400
    assert request.reason == "'id' parameter is required."

    # Unknown event.
    request = client.fetch("/event?id=bogus")
    assert request.code == 404
    assert request.reason == "Event not found."

    # Known event.
    request = client.fetch("/event?id=B071791B")
    assert request.code == 200
    event = json.loads(str(request.body.decode("utf8")))

    assert event == {
        "m_rr": -58000000000000000,
        "m_tt": 78100000000000000,
        "m_pp": -20100000000000000,
        "m_rt": -56500000000000000,
        "m_rp": 108100000000000000,
        "m_tp": 315300000000000000,
        "latitude": -3.8,
        "longitude": -104.21,
        "depth_in_m": 0,
        "origin_time": "1991-07-17T16:41:33.100000Z"}


def test_station_query_various_failures(
        all_clients_station_coordinates_callback):
    """
    The station query can fail for various reasons.
    """
    client = all_clients_station_coordinates_callback

    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000"}

    # It fails if receiver coordinates and query params are passed.
    p = copy.deepcopy(params)
    p["network"] = "IU,B*"
    p["station"] = "ANT*,ANM?"
    p["receiverlatitude"] = 1.0

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == (
        "Receiver coordinates can either be specified by passing "
        "the coordinates, or by specifying query parameters, "
        "but not both.")

    # It also fails if only one part is given.
    p = copy.deepcopy(params)
    p["network"] = "IU,B*"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == (
        "Must specify a full set of coordinates or a full set of receiver "
        "parameters.")

    # It also fails if it does not find any networks and stations.
    p = copy.deepcopy(params)
    p["network"] = "X*"
    p["station"] = "Y*"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 404
    assert request.reason == ("No coordinates found satisfying the query.")


def test_station_query_no_callback(all_clients):
    """
    Test the error message when no station callback is available.
    """
    client = all_clients

    params = {
        "sourcelatitude": 10, "sourcelongitude": 10, "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000"}

    p = copy.deepcopy(params)
    p["network"] = "IU,B*"
    p["station"] = "ANT*,ANM?"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support station coordinates and thus no station "
        "queries.")


def test_event_query_no_callbacks(all_clients):
    """
    Test the error message when no event callback is available.
    """
    client = all_clients

    params = {"receiverlatitude": 10, "receiverlongitude": 10}

    p = copy.deepcopy(params)
    p["eventid"] = "B071791B"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support event information and thus no event queries.")


def test_event_query_various_failures(all_clients_event_callback):
    """
    Various failure states of the eventid queries.
    """
    client = all_clients_event_callback

    params = {"receiverlatitude": 10, "receiverlongitude": 10}

    p = copy.deepcopy(params)
    p["eventid"] = "B071791B"
    p["sourcemomenttensor"] = "1,1,1,1,1,1"
    p["sourcelatitude"] = -20
    p["sourcelongitude"] = -20
    p["sourcedepthinmeters"] = -20

    # Cannot not pass other source parameters along.
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == (
        "The following parameters cannot be used if 'eventid' is a "
        "parameter: 'sourcedepthinmeters', 'sourcelatitude', "
        "'sourcelongitude', 'sourcemomenttensor'")

    # Neither can the origin time be specified.
    p = copy.deepcopy(params)
    p["eventid"] = "B071791B"
    p["origintime"] = obspy.UTCDateTime(2014, 1, 1)

    # Cannot not pass other source parameters along.
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == (
        "'eventid' and 'origintime' parameters cannot both be passed at the "
        "same time.")


def test_event_parameters_by_querying(all_clients_event_callback):
    """
    Test the query by eventid.
    """
    client = all_clients_event_callback

    # Only works for reciprocal databases. Otherwise the depth if fixed.
    if not client.is_reciprocal:
        return

    db = instaseis.open_db(client.filepath)

    params = {"receiverlatitude": 10, "receiverlongitude": 10,
              "format": "miniseed"}

    p = copy.deepcopy(params)
    p["eventid"] = "B071791B"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    st = obspy.read(request.buffer)

    # Test it against a manually queried seismogram.
    source = instaseis.Source(
        latitude=-3.8,
        longitude=-104.21,
        depth_in_m=0,
        m_pp=-20100000000000000,
        m_rt=-56500000000000000,
        m_rp=108100000000000000,
        m_rr=-58000000000000000,
        m_tp=315300000000000000,
        m_tt=78100000000000000,
        origin_time=obspy.UTCDateTime("1991-07-17T16:41:33.100000Z"))
    receiver = instaseis.Receiver(latitude=10, longitude=10, depth_in_m=0.0,
                                  network="XX", station="SYN")

    st_db = db.get_seismograms(source=source, receiver=receiver)

    for tr, tr_db in zip(st, st_db):
        del tr.stats._format
        del tr.stats.mseed
        del tr_db.stats.instaseis

        np.testing.assert_allclose([tr.stats.delta], [tr_db.stats.delta])
        tr.stats.delta = tr_db.stats.delta
        assert tr.stats == tr_db.stats
        np.testing.assert_allclose(tr.data, tr_db.data,
                                   atol=tr.data.ptp() / 1E9)

    # Also perform a mock comparison to test the actually created object.
    with mock.patch("instaseis.instaseis_db.InstaseisDB.get_seismograms") \
            as patch:
        _st = obspy.read()
        for tr in _st:
            tr.stats.starttime = source.origin_time - 0.01
            tr.stats.delta = 10.0
        patch.return_value = _st
        result = client.fetch(_assemble_url('seismograms', **p))
        assert result.code == 200

    assert patch.call_args[1]["source"] == source


def test_event_query_seismogram_non_existent_event(all_clients_event_callback):
    """
    Tests querying for an event that is not found.
    """
    client = all_clients_event_callback

    params = {"receiverlatitude": 10, "receiverlongitude": 10,
              "eventid": "bogus"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 404
    assert request.reason == "Event not found."


def test_label_parameter(all_clients):
    """
    Test the 'label' parameter of the /seismograms route.
    """
    prefix = "attachment; filename="
    client = all_clients

    params = {
        "sourcelatitude": 10, "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "receiverlatitude": 20, "receiverlongitude": 20, "format": "miniseed"}

    # No specified label will result in it having a generic label.
    p = copy.deepcopy(params)

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix):]
    assert filename.startswith("instaseis_seismogram_")
    assert filename.endswith(".mseed")

    # The same is true if saczip is used but in that case the ending is zip
    # and all the files inside have the trace id as the name.
    p = copy.deepcopy(params)
    p["format"] = "saczip"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix):]
    assert filename.startswith("instaseis_seismogram_")
    assert filename.endswith(".zip")

    zip_obj = zipfile.ZipFile(request.buffer)
    names = zip_obj.namelist()
    zip_obj.close()

    assert sorted(names) == sorted(
        ["XX.SYN..LXZ.sac", "XX.SYN..LXN.sac", "XX.SYN..LXE.sac"])

    # Now pass one. It will replace the filename prefix.
    p = copy.deepcopy(params)
    p["label"] = "Tohoku"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix):]
    assert filename.startswith("Tohoku_")
    assert filename.endswith(".mseed")

    # Same for saczip and also the files in the zip should change.
    p = copy.deepcopy(params)
    p["format"] = "saczip"
    p["label"] = "Tohoku"

    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix):]
    assert filename.startswith("Tohoku_")
    assert filename.endswith(".zip")

    zip_obj = zipfile.ZipFile(request.buffer)
    names = zip_obj.namelist()
    zip_obj.close()

    assert sorted(names) == sorted([
        "Tohoku_XX.SYN..LXZ.sac",
        "Tohoku_XX.SYN..LXN.sac",
        "Tohoku_XX.SYN..LXE.sac"])


def test_ttimes_route_no_callback(all_clients):
    """
    Tests the ttimes route with no available callbacks.
    """
    client = all_clients

    request = client.fetch(
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phase=P")
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations.")


def test_ttimes_route(all_clients_ttimes_callback):
    """
    Test for the ttimes route.
    """
    client = all_clients_ttimes_callback

    # Test with missing parameters.
    request = client.fetch(
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90")
    assert request.code == 400
    assert request.reason == (
        "The following required parameters are missing: "
        "'phase', 'receiverdepthinmeters'")

    # Invalid phase name
    request = client.fetch(
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phase=bogs")
    assert request.code == 400
    assert request.reason == "Invalid phase name."

    # Other error, e.g. negative depth.
    request = client.fetch(
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=-200&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phase=P")
    assert request.code == 400
    assert request.reason == ("Failed to calculate travel time due to: No "
                              "layer contains this depth")

    # No such phase at that distance.
    request = client.fetch(
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phase=Pdiff")
    assert request.code == 404
    assert request.reason == "No ray for the given geometry and phase found."

    # Many implementations will not have a receiverdepth. This one does not.
    request = client.fetch(
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=20&phase=Pdiff")
    assert request.code == 400
    assert request.reason == ("Failed to calculate travel time due to: This "
                              "travel time implementation cannot calculate "
                              "buried receivers.")

    # Last but not least test some actual travel times.
    request = client.fetch(
        "/ttimes?sourcelatitude=0&sourcelongitude=0&"
        "sourcedepthinmeters=300000&receiverlatitude=0&receiverlongitude=50&"
        "receiverdepthinmeters=0&phase=P")
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert list(result.keys()) == ["travel_time"]
    assert abs(result["travel_time"] - 504.357 < 1E-2)

    request = client.fetch(
        "/ttimes?sourcelatitude=0&sourcelongitude=0&"
        "sourcedepthinmeters=300000&receiverlatitude=0&receiverlongitude=50&"
        "receiverdepthinmeters=0&phase=PP")
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert list(result.keys()) == ["travel_time"]
    assert abs(result["travel_time"] - 622.559 < 1E-2)

    request = client.fetch(
        "/ttimes?sourcelatitude=0&sourcelongitude=0&"
        "sourcedepthinmeters=300000&receiverlatitude=0&receiverlongitude=50&"
        "receiverdepthinmeters=0&phase=sPKiKP")
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert list(result.keys()) == ["travel_time"]
    assert abs(result["travel_time"] - 1090.081 < 1E-2)


def test_network_and_station_code_settings(all_clients):
    """
    Tests the network and station code settings.
    """
    client = all_clients

    params = {
        "sourcelatitude": 10, "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "receiverlatitude": 20, "receiverlongitude": 20, "format": "miniseed"}

    # Default network and station codes.
    p = copy.deepcopy(params)
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "XX"
        assert tr.stats.station == "SYN"

    # Set only the network code.
    p = copy.deepcopy(params)
    p["networkcode"] = "BW"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "BW"
        assert tr.stats.station == "SYN"

    # Set only the station code.
    p = copy.deepcopy(params)
    p["stationcode"] = "INS"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "XX"
        assert tr.stats.station == "INS"

    # Set both.
    p = copy.deepcopy(params)
    p["networkcode"] = "BW"
    p["stationcode"] = "INS"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "BW"
        assert tr.stats.station == "INS"

    # Station code is limited to five letters.
    p = copy.deepcopy(params)
    p["stationcode"] = "123456"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == "'stationcode' must have 5 or fewer letters."

    # Network code is limited to two letters.
    p = copy.deepcopy(params)
    p["networkcode"] = "123"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == "'networkcode' must have 2 or fewer letters."


def test_phase_relative_offsets(all_clients_ttimes_callback):
    """
    Test phase relative offsets.

    + must be encoded with %2B
    - must be encoded with %2D
    """
    client = all_clients_ttimes_callback

    # Only for reciprocal ones as the depth is fixed otherwise.
    if not client.is_reciprocal:
        return

    # At a distance of 50 degrees and with a source depth of 300 km:
    # P: 504.357 seconds
    # PP: 622.559 seconds
    # sPKiKP: 1090.081 seconds

    params = {
        "sourcelatitude": 0, "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0, "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed"}

    # Normal seismogram.
    p = copy.deepcopy(params)
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    starttime, endtime = tr.stats.starttime, tr.stats.endtime

    # Start 10 seconds before the P arrival.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Starts 10 seconds after the P arrival
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Ends 15 seconds before the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2D15"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 15)) < 0.1

    # Ends 15 seconds after the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2B15"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1

    # Starts 5 seconds before the PP and ends 2 seconds after the sPKiKP phase.
    p = copy.deepcopy(params)
    p["starttime"] = "PP%2D5"
    p["endtime"] = "sPKiKP%2B2"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 622.559 - 5)) < 0.1
    assert abs((tr.stats.endtime) - (starttime + 1090.081 + 2)) < 0.1

    # Combinations with relative end times are also possible. Relative start
    # times are always relative to the origin time so it does not matter in
    # that case.
    p = copy.deepcopy(params)
    p["starttime"] = "PP%2D5"
    p["endtime"] = 10.0
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 622.559 - 5)) < 0.1
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 5 + 10)) < 0.1

    # Nonetheless, also test the other combination of relative start time
    # and phase relative endtime.
    p = copy.deepcopy(params)
    p["starttime"] = "10"
    p["endtime"] = "PP%2B15"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime + 10
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1


def test_phase_relative_offsets_but_no_ttimes_callback(all_clients):
    client = all_clients

    params = {
        "sourcelatitude": 0, "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0, "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed"}

    # Test for starttime.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations.")

    # Test for endtime.
    p = copy.deepcopy(params)
    p["endtime"] = "P%2D10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations.")

    # Test for both.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    p["endtime"] = "S%2B10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations.")


def test_phase_relative_offset_different_time_representations(
        all_clients_ttimes_callback):
    client = all_clients_ttimes_callback

    # Different source depth for non-reciprocal client...
    if not client.is_reciprocal:
        return

    params = {
        "sourcelatitude": 0, "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0, "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed"}

    # Normal seismogram.
    p = copy.deepcopy(params)
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    starttime, endtime = tr.stats.starttime, tr.stats.endtime

    # P+10
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-10
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+10.0
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10.0"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-10.0
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10.0"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+10.000
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10.000"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-10.000
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10.000"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+1E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D1E1"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-1E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B1E1"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+1.0E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D1.0E1"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-1.0E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B1.0E1"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+1e1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D1e1"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-1e1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B1e1"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime


def test_phase_relative_offset_failures(all_clients_ttimes_callback):
    """
    Tests some common failures for the phase relative offsets.
    """
    client = all_clients_ttimes_callback

    params = {
        "sourcelatitude": 0, "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0, "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed"}

    # Illegal phase.
    p = copy.deepcopy(params)
    p["starttime"] = "bogus%2D10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == "Invalid phase name: bogus"

    # Phase not available at that distance.
    p = copy.deepcopy(params)
    p["starttime"] = "Pdiff%2D10"
    request = client.fetch(_assemble_url('seismograms', **p))
    assert request.code == 400
    assert request.reason == (
        "No seismograms found for the given phase relative "
        "offsets. This could either be due to the chosen phase "
        "not existing for the specific source-receiver geometry "
        "or arriving too late/with too large offsets if the "
        "database is not long enough.")


def test_phase_relative_offsets_multiple_stations(all_clients_all_callbacks):
    client = all_clients_all_callbacks

    # Only for reciprocal ones as the depth is different otherwise...
    if not client.is_reciprocal:
        return

    # Now test multiple receiveers.
    # This is constructed in such a way that only one station will have a P
    # phase (due to the distance). So this is tested here.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?",
        "starttime": "P%2D10"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert len(st) == 1

    # This is constructed in such a way that only one station will have a P
    # phase (due to the distance). So this is tested here.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?",
        "endtime": "P%2D10"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert len(st) == 1

    # Now get both.
    params = {
        "sourcelatitude": 39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?",
        "starttime": "P%2D10"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert len(st) == 2

    # Or one also does not get any. In that case an error is raised.
    params = {
        "sourcelatitude": 39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?",
        "starttime": "P%2D10000"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "No seismograms found for the given phase relative "
        "offsets. This could either be due to the chosen phase "
        "not existing for the specific source-receiver geometry "
        "or arriving too late/with too large offsets if the "
        "database is not long enough.")

    # Or one also does not get any. In that case an error is raised.
    params = {
        "sourcelatitude": 39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?",
        "endtime": "P%2B10000"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "No seismograms found for the given phase relative "
        "offsets. This could either be due to the chosen phase "
        "not existing for the specific source-receiver geometry "
        "or arriving too late/with too large offsets if the "
        "database is not long enough.")


def test_various_failure_conditions(all_clients_all_callbacks):
    client = all_clients_all_callbacks
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    # no source mechanism given.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "One of the following has to be given: 'eventid', "
        "'sourcedoublecouple', 'sourceforce', 'sourcemomenttensor'")

    # moment tensor is missing a component.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcemomenttensor' must be formatted as: "
        "'Mrr,Mtt,Mpp,Mrt,Mrp,Mtp'")

    # moment tensor has an invalid component.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,bogus",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcemomenttensor' must be formatted as: "
        "'Mrr,Mtt,Mpp,Mrt,Mrp,Mtp'")

    # sourcedoublecouple is missing a component.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcedoublecouple' must be formatted as: "
        "'strike,dip,rake[,M0]'")

    # sourcedoublecouple has an extra component.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,11,12",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcedoublecouple' must be formatted as: "
        "'strike,dip,rake[,M0]'")

    # sourcedoublecouple has an invalid component.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,bogus",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcedoublecouple' must be formatted as: "
        "'strike,dip,rake[,M0]'")

    # sourceforce is missing a component.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourceforce": "100000,100000",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourceforce' must be formatted as: "
        "'Fr,Ft,Fp'")

    # sourceforce has an invalid component.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourceforce": "100000,100000,bogus",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourceforce' must be formatted as: "
        "'Fr,Ft,Fp'")

    # Seismic moment cannot be negative.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,-10",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == "Seismic moment must not be negative."

    # Funky phase offset setting.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,10",
        "components": "Z", "dt": 0.1,
        "starttime": "P+!A",
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'starttime' must be formatted as: 'Datetime "
        "String/Float/Phase+-Offset'")

    # Mixing different source settings.
    params = {
        "sourcelatitude": -39, "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,10",
        "sourceforce": "10,10,10",
        "components": "Z", "dt": 0.1,
        "format": "miniseed", "network": "IU,B*", "station": "ANT*,ANM?"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == ("Only one of these parameters can be given "
                              "simultaneously: 'sourcedoublecouple', "
                              "'sourceforce'")

    if db.info.is_reciprocal:
        # Receiver depth must be at the surface for a reciprocal database.
        params = {
            "sourcelatitude": -39, "sourcelongitude": 20,
            "sourcedepthinmeters": 0,
            "sourcedoublecouple": "10,10,10,10",
            "receiverlatitude": 10, "receiverlongitude": 10,
            "receiverdepthinmeters": 10,
            "components": "Z", "dt": 0.1,
            "format": "miniseed"}
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 400
        assert request.reason == ("Receiver must be at the surface for "
                                  "reciprocal databases.")

        # A too deep source depth raises.
        params = {
            "sourcelatitude": -39, "sourcelongitude": 20,
            "sourcedepthinmeters": 3E9,
            "sourcedoublecouple": "10,10,10,10",
            "receiverlatitude": 10, "receiverlongitude": 10,
            "components": "Z", "dt": 0.1,
            "format": "miniseed"}
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 400
        assert request.reason == ("Source depth must be within the database "
                                  "range: 0.0 - 371000.0 meters.")

    # For a forward database the source depth must be equal to the database
    # depth!
    if db.info.is_reciprocal is False:
        params = {
            "sourcelatitude": -39, "sourcelongitude": 20,
            "sourcedepthinmeters": 14,
            "sourcedoublecouple": "10,10,10,10",
            "receiverlatitude": 10, "receiverlongitude": 10,
            "receiverdepthinmeters": 10,
            "components": "Z", "dt": 0.1,
            "format": "miniseed"}
        request = client.fetch(_assemble_url('seismograms', **params))
        assert request.code == 400
        assert request.reason == (
            "Source depth must be: %.1f km" % db.info.source_depth)


def test_sac_headers(all_clients):
    """
    Tests the sac headers.
    """
    client = all_clients

    params = {
        "sourcelatitude": 1, "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "origintime": obspy.UTCDateTime(0),
        "dt": 0.1, "starttime": "-1.5", "receiverlatitude": 22,
        "receiverlongitude": 44, "format": "saczip"}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200
    st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st:
        assert tr.stats._format == "SAC"
        # Instaseis will write SAC coordinates in WGS84!
        # Assert the station headers.
        assert abs(tr.stats.sac.stla -
                   geocentric_to_elliptic_latitude(22)) < 1E-6
        assert abs(tr.stats.sac.stlo - 44) < 1E-6
        assert abs(tr.stats.sac.stdp - 0.0) < 1E-6
        assert abs(tr.stats.sac.stel - 0.0) < 1E-6
        # Assert the event parameters.
        assert abs(tr.stats.sac.evla -
                   geocentric_to_elliptic_latitude(1)) < 1E-6
        assert abs(tr.stats.sac.evlo - 12) < 1E-6
        assert abs(tr.stats.sac.evdp - client.source_depth) < 1E-6
        assert abs(tr.stats.sac.mag - 4.22) < 1E-2
        # Thats what SPECFEM uses for a moment magnitude....
        assert tr.stats.sac.imagtyp == 55
        # Assume the reference time is the starttime.
        assert abs(tr.stats.sac.o - 1.5) < 1E-6
        # Test the provenance.
        assert tr.stats.sac.kuser0 == "InstSeis"
        assert tr.stats.sac.kuser1 == instaseis.__version__[:8]
        assert tr.stats.sac.kuser2 == "prem_iso"


def test_dt_settings(all_clients):
    """
    Cannot downsample nor sample to more than 100 Hz.
    """
    client = all_clients

    # Requesting exactly at the initial sampling rate works.
    params = {
        "sourcelatitude": 1, "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": client.info.dt,
        "receiverlatitude": 22, "receiverlongitude": 44}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200

    # Request exactly at 100 Hz works.
    params = {
        "sourcelatitude": 1, "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": 0.01,
        "receiverlatitude": 22, "receiverlongitude": 44}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200

    # Requesting at something in between works.
    params = {
        "sourcelatitude": 1, "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": 0.123,
        "receiverlatitude": 22, "receiverlongitude": 44}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 200

    # Requesting a tiny bit above does not work.
    params = {
        "sourcelatitude": 1, "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": 0.009,
        "receiverlatitude": 22, "receiverlongitude": 44}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "The smallest possible dt is 0.01. Please choose a smaller value and "
        "resample locally if needed.")

    # Requesting a tiny bit below also does not work.
    params = {
        "sourcelatitude": 1, "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": client.info.dt + 0.001,
        "receiverlatitude": 22, "receiverlongitude": 44}
    request = client.fetch(_assemble_url('seismograms', **params))
    assert request.code == 400
    assert request.reason == (
        "Cannot downsample. The sampling interval of the database is "
        "24.72485 seconds. Make sure to choose a smaller or equal one.")

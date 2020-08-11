#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Instaseis server.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import copy
import io
import json
import zipfile

import obspy
import numpy as np
from scipy.integrate import simps
import pytest
from .tornado_testing_fixtures import *  # NOQA
from .tornado_testing_fixtures import _assemble_url

import instaseis
from instaseis.helpers import geocentric_to_elliptic_latitude
from instaseis.server import util

# Conditionally import mock either from the stdlib or as a separate library.
import sys

if sys.version_info[0] == 2:  # pragma: no cover
    import mock
else:  # pragma: no cover
    import unittest.mock as mock


def _compare_streams(st1, st2):
    for tr1, tr2 in zip(st1, st2):
        assert tr1.stats.__dict__ == tr2.stats.__dict__
        rtol = 1e-3
        atol = 1e-4 * max(np.abs(tr1.data).max(), np.abs(tr2.data).max())
        np.testing.assert_allclose(tr1.data, tr2.data, rtol=rtol, atol=atol)


def fetch_sync(client, url, **kwargs):
    """
    Helper function to call an async test client in a sync test case.
    """

    async def f():
        try:
            response = await client.fetch(
                f"http://localhost:{client.port}{url}", **kwargs
            )
        except Exception as e:
            response = e.response
        return response

    return client.io_loop.run_sync(f)


def test_root_route(all_clients):
    """
    Shows very basic information and the version of the client. Test is run
    for all clients.
    """
    client = all_clients
    request = fetch_sync(client, "/")
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert result == {
        "type": "Instaseis Remote Server",
        "version": instaseis.__version__,
    }
    assert request.headers["Content-Type"] == "application/json; charset=UTF-8"


def test_info_route(all_clients):
    """
    Tests that the /info route returns the information dictionary and does
    not mess with anything.

    Test is parameterized to run for all test databases.
    """
    client = all_clients
    # Load the result via the webclient.
    request = fetch_sync(client, "/info")
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


def test_greens_function_error_handling_no_reciprocal_db(all_clients):
    """
    Tests the error the greens route gives if the database is not reciprocal.
    """
    client = all_clients

    params = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "saczip",
    }
    request = fetch_sync(client, _assemble_url("greens_function", **params))

    if (
        client.is_reciprocal
        and client.info.components == "vertical and horizontal"
    ):
        assert request.code == 200
    elif client.info.components == "4 elemental moment tensors":
        assert request.code == 400
        assert request.reason == (
            "The database is not reciprocal, so Green's "
            "functions can't be computed."
        )
    else:
        assert request.code == 400
        assert request.reason == (
            "Database requires vertical AND horizontal "
            "components to be able to compute Green's "
            "functions."
        )


def test_greens_function_error_handling(all_greens_clients):
    """
    Tests error handling of the /greens_function route. Very basic for now
    """
    client = all_greens_clients

    basic_parameters = {
        "sourcedepthinmeters": client.source_depth,
        "sourcedistanceindegrees": 20,
    }

    # Remove the sourcedistanceindegrees, required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcedistanceindegrees"]
    request = fetch_sync(client, _assemble_url("greens_function", **params))
    assert request.code == 400
    assert (
        request.reason
        == "Required parameter 'sourcedistanceindegrees' not given."
    )

    # Remove the sourcedepthinmeters, required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcedepthinmeters"]
    request = fetch_sync(client, _assemble_url("greens_function", **params))
    assert request.code == 400
    assert (
        request.reason == "Required parameter 'sourcedepthinmeters' not given."
    )

    # Negative source distance
    request = fetch_sync(
        client,
        _assemble_url(
            "greens_function",
            sourcedepthinmeters=20,
            sourcedistanceindegrees=-30,
        ),
    )
    assert request.code == 400
    assert request.reason == "Epicentral distance should be in [0, 180]."

    # Too far source distances.
    request = fetch_sync(
        client,
        _assemble_url(
            "greens_function",
            sourcedepthinmeters=20,
            sourcedistanceindegrees=200,
        ),
    )
    assert request.code == 400
    assert request.reason == "Epicentral distance should be in [0, 180]."

    # Negative source depth.
    request = fetch_sync(
        client,
        _assemble_url(
            "greens_function",
            sourcedepthinmeters=-20,
            sourcedistanceindegrees=20,
        ),
    )
    assert request.code == 400
    assert request.reason == "Source depth should be in [0.0, 371000.0]."

    # Too large source depth.
    request = fetch_sync(
        client,
        _assemble_url(
            "greens_function",
            sourcedepthinmeters=2e6,
            sourcedistanceindegrees=20,
        ),
    )
    assert request.code == 400
    assert request.reason == "Source depth should be in [0.0, 371000.0]."


def test_greens_function_retrieval(all_greens_clients):
    """
    Tests if the greens functions requested from the server are identical to
    the one requested with the local instaseis client.
    """
    client = all_greens_clients

    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "saczip",
    }

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # default parameters
    params = copy.deepcopy(basic_parameters)
    params["origintime"] = str(time)
    request = fetch_sync(client, _assemble_url("greens_function", **params))
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
        epicentral_distance_in_degree=params["sourcedistanceindegrees"],
        source_depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        definition="seiscomp",
    )
    for tr in st_db:
        tr.stats.network = "XX"
        tr.stats.station = "GF001"

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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # miniseed
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    params["origintime"] = str(time)
    request = fetch_sync(client, _assemble_url("greens_function", **params))
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    for tr in st_server:
        assert tr.stats._format == "MSEED"

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params["sourcedistanceindegrees"],
        source_depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        definition="seiscomp",
    )
    for tr in st_db:
        tr.stats.network = "XX"
        tr.stats.station = "GF001"

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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # One with a label.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    params["label"] = "random_things"
    params["origintime"] = str(time)
    request = fetch_sync(client, _assemble_url("greens_function", **params))
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
    params["origintime"] = str(time)
    request = fetch_sync(client, _assemble_url("greens_function", **params))
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params["sourcedistanceindegrees"],
        source_depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        definition="seiscomp",
        dt=0.1,
        kernelwidth=2,
        kind="acceleration",
    )
    for tr in st_db:
        tr.stats.network = "XX"
        tr.stats.station = "GF001"

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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # One simulating a crash in the underlying function.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch(
        "instaseis.database_interfaces.base_instaseis_db"
        ".BaseInstaseisDB.get_greens_function"
    ) as p:

        p.side_effect = ValueError("random crash")
        request = fetch_sync(
            client, _assemble_url("greens_function", **params)
        )

    assert request.code == 400
    assert request.reason == (
        "Could not extract Green's function. Make "
        "sure, the parameters are valid, and the depth "
        "settings are correct."
    )

    # Two more simulating logic erros that should not be able to happen.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch(
        "instaseis.database_interfaces.base_instaseis_db"
        ".BaseInstaseisDB.get_greens_function"
    ) as p:
        st = obspy.read()
        for tr in st:
            tr.stats.starttime = obspy.UTCDateTime(1e5)

        p.return_value = st
        request = fetch_sync(
            client, _assemble_url("greens_function", **params)
        )

    assert request.code == 500
    assert request.reason == (
        "Starttime more than one hour before the "
        "starttime of the seismograms."
    )

    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch(
        "instaseis.database_interfaces.base_instaseis_db"
        ".BaseInstaseisDB.get_greens_function"
    ) as p:
        st = obspy.read()
        for tr in st:
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1)
            tr.stats.delta = 0.0001

        p.return_value = st
        request = fetch_sync(
            client, _assemble_url("greens_function", **params)
        )

    assert request.code == 500
    assert request.reason.startswith(
        "Endtime larger than the extracted " "endtime"
    )


def test_phase_relative_offsets_but_no_ttimes_callback_greens_function(
    all_greens_clients,
):
    client = all_greens_clients

    params = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "miniseed",
    }

    # Test for starttime.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations."
    )

    # Test for endtime.
    p = copy.deepcopy(params)
    p["endtime"] = "P%2D10"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations."
    )

    # Test for both.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    p["endtime"] = "S%2B10"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations."
    )


def test_phase_relative_offset_failures_greens_function(
    all_greens_clients_ttimes_callback,
):
    """
    Tests some common failures for the phase relative offsets with the
    greens function route.
    """
    client = all_greens_clients_ttimes_callback

    params = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "miniseed",
    }

    # Illegal phase.
    p = copy.deepcopy(params)
    p["starttime"] = "bogus%2D10"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 400
    assert request.reason == "Invalid phase name: bogus"

    # Phase not available at that distance.
    p = copy.deepcopy(params)
    p["starttime"] = "Pdiff%2D10"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 400
    assert request.reason == (
        "No Green's function extracted for the given phase relative offsets. "
        "This could either be due to the chosen phase not existing for the "
        "specific source-receiver geometry or arriving too late/with too "
        "large offsets if the database is not long enough."
    )


def test_phase_relative_offsets_greens_function(
    all_greens_clients_ttimes_callback,
):
    """
    Test phase relative offsets with the green's function route.

    + must be encoded with %2B
    - must be encoded with %2D
    """
    client = all_greens_clients_ttimes_callback

    # At a distance of 50 degrees and with a source depth of 300 km:
    # P: 504.357 seconds
    # PP: 622.559 seconds
    # sPKiKP: 1090.081 seconds
    params = {
        "sourcedepthinmeters": 300000,
        "sourcedistanceindegrees": 50,
        "format": "miniseed",
        "dt": 0.1,
    }

    # Normal seismogram.
    p = copy.deepcopy(params)
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    starttime, endtime = tr.stats.starttime, tr.stats.endtime

    # Start 10 seconds before the P arrival.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Starts 10 seconds after the P arrival
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Ends 15 seconds before the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2D15"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 15)) < 0.1

    # Ends 15 seconds after the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2B15"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1

    # Starts 5 seconds before the PP and ends 2 seconds after the sPKiKP phase.
    p = copy.deepcopy(params)
    p["starttime"] = "PP%2D5"
    p["endtime"] = "sPKiKP%2B2"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
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
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 622.559 - 5)) < 0.1
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 5 + 10)) < 0.1

    # Nonetheless, also test the other combination of relative start time
    # and phase relative endtime.
    p = copy.deepcopy(params)
    p["starttime"] = "10"
    p["endtime"] = "PP%2B15"
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime + 10
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1


def test_greens_function_start_and_origintime(all_greens_clients):
    client = all_greens_clients

    basic_parameters = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "miniseed",
    }

    # default parameters
    params = copy.deepcopy(basic_parameters)
    request = fetch_sync(client, _assemble_url("greens_function", **params))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert st[0].stats.starttime == obspy.UTCDateTime(1900, 1, 1)

    # Just setting the origin time.
    time = obspy.UTCDateTime(2016, 1, 1)
    p = copy.deepcopy(basic_parameters)
    p["origintime"] = str(time)
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert st[0].stats.starttime == time

    # Setting it to something early.
    time = obspy.UTCDateTime(1990, 1, 1)
    p = copy.deepcopy(basic_parameters)
    p["origintime"] = str(time)
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert st[0].stats.starttime == time

    # Setting it to something early.
    time = obspy.UTCDateTime(1990, 1, 1)
    p = copy.deepcopy(basic_parameters)
    p["origintime"] = str(time)
    p["starttime"] = str(time - 60 * 30)
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    st_new = obspy.read(request.buffer)
    # This should be about 30 mins.
    assert abs(st_new[0].stats.starttime - (time - 60 * 30)) < 60
    # The endtime should not change compared to the previous entry. Floating
    # point math and funny sampling rates result in some inaccuracies.
    assert abs(st_new[0].stats.endtime - st[0].stats.endtime) < 0.01

    # Also check the endtime.
    time = obspy.UTCDateTime(1990, 1, 1)
    p = copy.deepcopy(basic_parameters)
    p["origintime"] = str(time)
    p["starttime"] = str(time - 60 * 30)
    p["endtime"] = str(time + 20 * 60)
    request = fetch_sync(client, _assemble_url("greens_function", **p))
    assert request.code == 200
    st_new = obspy.read(request.buffer)
    # This should be about 30 mins.
    assert abs(st_new[0].stats.starttime - (time - 60 * 30)) < 60
    assert abs(st_new[0].stats.endtime - (time + 60 * 20)) < 60


def test_raw_seismograms_error_handling(all_clients):
    """
    Tests error handling of the /seismograms_raw route. Potentially outwards
    facing thus tested rather well.
    """
    client = all_clients

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverdepthinmeters": 0,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
    }

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcelatitude"]
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 400
    assert request.reason == "Required parameter 'sourcelatitude' not given."

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["sourcelatitude"] = "A"
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "sourcelatitude" in request.reason

    # No source.
    request = fetch_sync(
        client, _assemble_url("seismograms_raw", **basic_parameters)
    )
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
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
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
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["strike"] = "45"
    params["dip"] = "45"
    params["rake"] = "45"
    params["M0"] = "450000"
    params["sourcelatitude"] = "100"
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
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
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
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
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()

    # Unlikely to be raised for real, but test the resulting error nonetheless.
    with mock.patch(
        "instaseis.database_interfaces.base_instaseis_db"
        ".BaseInstaseisDB._convert_to_stream"
    ) as p:
        p.side_effect = Exception

        params = copy.deepcopy(basic_parameters)
        params["mtt"] = "100000"
        params["mpp"] = "100000"
        params["mrr"] = "100000"
        params["mrt"] = "100000"
        params["mrp"] = "100000"
        params["mtp"] = "100000"
        request = fetch_sync(
            client, _assemble_url("seismograms_raw", **params)
        )
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
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
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
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 400
    assert "a request with no components will not re" in request.reason.lower()


def test_seismograms_raw_route(all_clients):
    """
    Test the raw routes. Make sure the response is a MiniSEED file with the
    correct channels.

    Once again executed for each known test database.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
    }

    # Various sources.
    mt = {
        "mtt": "100000",
        "mpp": "200000",
        "mrr": "300000",
        "mrt": "400000",
        "mrp": "500000",
        "mtp": "600000",
    }
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == len(db.default_components)

    # Assert the MiniSEED file and some basic properties.
    for tr in st:
        assert hasattr(tr.stats, "mseed")
        assert tr.data.dtype.char == "f"

    # Strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params.update(sdr)
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == len(db.default_components)

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
        request = fetch_sync(
            client, _assemble_url("seismograms_raw", **params)
        )
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
        if not all([_i in db.available_components for _i in comp]):
            continue
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        params["components"] = comp
        request = fetch_sync(
            client, _assemble_url("seismograms_raw", **params)
        )
        assert request.code == 200

        st = obspy.read(request.buffer)
        assert len(st) == len(comp)
        assert "".join(sorted(comp)) == "".join(
            sorted([tr.stats.channel[-1] for tr in st])
        )

    # Test passing the origin time.
    params = copy.deepcopy(basic_parameters)
    time = obspy.UTCDateTime(2013, 1, 2, 3, 4, 5)
    params.update(mt)
    params["origintime"] = str(time)
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == len(db.default_components)
    for tr in st:
        assert tr.stats.starttime == time

    # Test passing network and station codes.
    params = copy.deepcopy(basic_parameters)
    params.update(mt)
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"
    params["locationcode"] = "XX"
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 200

    st = obspy.read(request.buffer)
    assert len(st) == len(db.default_components)
    for tr in st:
        assert tr.stats.network == "BW"
        assert tr.stats.station == "ALTM"
        assert tr.stats.location == "XX"


def test_mu_is_passed_as_header_value(all_clients):
    """
    Makes sure mu is passed as a header value.

    Also tests the other headers.
    """
    client = all_clients
    parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "mtt": "100000",
        "mpp": "100000",
        "mrr": "100000",
        "mrt": "100000",
        "mrp": "100000",
        "mtp": "100000",
    }

    # Moment tensor source.
    request = fetch_sync(
        client, _assemble_url("seismograms_raw", **parameters)
    )
    assert request.code == 200
    # Make sure the mu header exists and the value can be converted to a float.
    assert "Instaseis-Mu" in request.headers
    assert isinstance(float(request.headers["Instaseis-Mu"]), float)

    assert request.headers["Content-Type"] == "application/vnd.fdsn.mseed"
    cd = request.headers["Content-Disposition"]
    assert "attachment; filename=" in cd
    assert "instaseis_seismogram" in cd


def test_object_creation_for_raw_seismogram_route(all_clients):
    """
    Tests that the correct objects are created for the raw seismogram route.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
    }

    # Various sources.
    mt = {
        "mtt": "100000",
        "mpp": "200000",
        "mrr": "300000",
        "mrt": "400000",
        "mrp": "500000",
        "mtp": "600000",
    }
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    with mock.patch(
        "instaseis.database_interfaces.base_netcdf_instaseis_db"
        ".BaseNetCDFInstaseisDB._get_seismograms"
    ) as p:
        _st = obspy.read()
        for tr in _st:
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = obspy.UTCDateTime(0)
        data = {}
        data["mu"] = 1.0
        for tr in _st:
            data[tr.stats.channel[-1]] = tr.data
        p.return_value = data

        # Moment tensor source.
        params = copy.deepcopy(basic_parameters)
        params.update(mt)
        request = fetch_sync(
            client, _assemble_url("seismograms_raw", **params)
        )
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            **dict(
                (key[0] + "_" + key[1:], float(value))
                for (key, value) in mt.items()
            ),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0,
        )

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["sourcedepthinmeters"] = "5.0"
        params["origintime"] = str(time)
        params["receiverdepthinmeters"] = "55.0"
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"
        params["locationcode"] = "XX"

        request = fetch_sync(
            client, _assemble_url("seismograms_raw", **params)
        )
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=5.0,
            origin_time=time,
            **dict(
                (key[0] + "_" + key[1:], float(value))
                for (key, value) in mt.items()
            ),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=55.0,
            network="BW",
            station="ALTM",
            location="XX",
        )

        # From strike, dip, rake
        p.reset_mock()

        params = copy.deepcopy(basic_parameters)
        params.update(sdr)
        request = fetch_sync(
            client, _assemble_url("seismograms_raw", **params)
        )
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1][
            "source"
        ] == instaseis.Source.from_strike_dip_rake(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            **dict((key, float(value)) for (key, value) in sdr.items()),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0,
        )

        # Moment tensor source with a couple more parameters.
        p.reset_mock()

        params["sourcedepthinmeters"] = "5.0"
        params["origintime"] = str(time)
        params["receiverdepthinmeters"] = "55.0"
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"
        params["locationcode"] = "XX"

        request = fetch_sync(
            client, _assemble_url("seismograms_raw", **params)
        )
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1][
            "source"
        ] == instaseis.Source.from_strike_dip_rake(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=5.0,
            origin_time=time,
            **dict((key, float(value)) for (key, value) in sdr.items()),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=55.0,
            network="BW",
            station="ALTM",
            location="XX",
        )

        # Force source only works for displ_only databases.
        if "displ_only" in client.filepath:
            p.reset_mock()

            params = copy.deepcopy(basic_parameters)
            params.update(fs)
            request = fetch_sync(
                client, _assemble_url("seismograms_raw", **params)
            )
            assert request.code == 200

            assert p.call_count == 1
            assert sorted(p.call_args[1]["components"]) == sorted(
                db.default_components
            )
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=0.0,
                **dict(
                    ("_".join(key), float(value))
                    for (key, value) in fs.items()
                ),
            )
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=0.0,
            )

            # Moment tensor source with a couple more parameters.
            p.reset_mock()

            params["sourcedepthinmeters"] = "5.0"
            params["origintime"] = str(time)
            params["receiverdepthinmeters"] = "55.0"
            params["networkcode"] = "BW"
            params["stationcode"] = "ALTM"
            params["locationcode"] = "XX"

            request = fetch_sync(
                client, _assemble_url("seismograms_raw", **params)
            )
            assert request.code == 200

            assert p.call_count == 1
            assert sorted(p.call_args[1]["components"]) == sorted(
                db.default_components
            )
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=5.0,
                origin_time=time,
                **dict(
                    ("_".join(key), float(value))
                    for (key, value) in fs.items()
                ),
            )
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=55.0,
                network="BW",
                station="ALTM",
                location="XX",
            )


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
        "receiverlongitude": -10,
    }

    # No source given at all.
    request = fetch_sync(
        client,
        _assemble_url(
            "seismograms", receiverlatitude=-10, receiverlongitude=10
        ),
    )
    assert request.code == 400
    assert request.reason == "No source specified"

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["sourcelatitude"]
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert (
        request.reason
        == "The following required parameters are missing: 'sourcelatitude'"
    )

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["sourcelatitude"] = "A"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "sourcelatitude" in request.reason

    # No source.
    request = fetch_sync(
        client, _assemble_url("seismograms", **basic_parameters)
    )
    assert request.code == 400
    assert request.reason == (
        "One of the following has to be given: 'eventid', "
        "'sourcedoublecouple', 'sourceforce', 'sourcemomenttensor'"
    )

    # Invalid receiver.
    params = copy.deepcopy(basic_parameters)
    params["receiverlatitude"] = "100"
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "could not construct receiver with " in request.reason.lower()

    # Invalid MT source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["sourcelatitude"] = "100"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = "45,45,45,450000"
    params["sourcelatitude"] = "100"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "could not construct the source" in request.reason.lower()
    assert "strike/dip/rake" in request.reason.lower()

    # Invalid force source. It only works in displ_only mode but here it
    # fails earlier.
    params = copy.deepcopy(basic_parameters)
    params["sourceforce"] = "100000,100000,100000"
    params["sourcelatitude"] = "100"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "could not construct force source" in request.reason.lower()

    # Could not extract seismogram.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["components"] = "ABC"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()

    # Wrong type of seismogram requested.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["units"] = "fun"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "unit must be one of" in request.reason.lower()

    # dt is too small - protects the server from having to serve humongous
    # files.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["dt"] = "0.009"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "the smallest possible dt is 0.01" in request.reason.lower()

    # interpolation kernel width is too wide or too narrow.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["kernelwidth"] = "0"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "`kernelwidth` must not be smaller" in request.reason.lower()
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["kernelwidth"] = "21"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "`kernelwidth` must not be smaller" in request.reason.lower()

    # too many components raise to avoid abuse.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["components"] = "NNEERRTTZZ"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "a maximum of 5 components can be request" in request.reason.lower()

    # At least one components must be requested.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = "100000,100000,100000,100000,100000,100000"
    params["components"] = ""
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "a request with no components will not re" in request.reason.lower()


def test_object_creation_for_seismogram_route(all_clients):
    """
    Tests that the correct objects are created for the seismogram route.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "receiverdepthinmeters": 0.0,
    }

    dt = 24.724845445855724

    # Various sources.
    mt = {
        "mrr": "100000",
        "mtt": "200000",
        "mpp": "300000",
        "mrt": "400000",
        "mrp": "500000",
        "mtp": "600000",
    }
    mt_param = "100000,200000,300000,400000,500000,600000"
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    sdr_param = "10,20,30,1000000"
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}
    fs_param = "100000,200000,300000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    with mock.patch(
        "instaseis.database_interfaces.base_netcdf_instaseis_db"
        ".BaseNetCDFInstaseisDB.get_seismograms"
    ) as p:
        _st = obspy.read()
        for tr in _st:
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt
        p.return_value = _st

        # Moment tensor source.
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=client.source_depth,
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(
                (key[0] + "_" + key[1:], float(value))
                for (key, value) in mt.items()
            ),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0,
            network="XX",
            station="SYN",
            location="SE",
        )
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
        params["locationcode"] = "XX"

        # We need to adjust the time values for the mock here.
        _st.traces = obspy.read().traces
        for tr in _st:
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = time - 1 - 7 * dt
            tr.stats.delta = dt

        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1]["source"] == instaseis.Source(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=params["sourcedepthinmeters"],
            origin_time=time,
            **dict(
                (key[0] + "_" + key[1:], float(value))
                for (key, value) in mt.items()
            ),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=params["receiverdepthinmeters"],
            network="BW",
            station="ALTM",
            location="XX",
        )
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
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt

        params = copy.deepcopy(basic_parameters)
        params["sourcedoublecouple"] = sdr_param
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1][
            "source"
        ] == instaseis.Source.from_strike_dip_rake(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=basic_parameters["sourcedepthinmeters"],
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict((key, float(value)) for (key, value) in sdr.items()),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0,
            network="XX",
            station="SYN",
            location="SE",
        )
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
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
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
        params["locationcode"] = "XX"

        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1][
            "source"
        ] == instaseis.Source.from_strike_dip_rake(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=params["sourcedepthinmeters"],
            origin_time=time,
            **dict((key, float(value)) for (key, value) in sdr.items()),
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=params["receiverdepthinmeters"],
            network="BW",
            station="ALTM",
            location="XX",
        )
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
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt

        params = copy.deepcopy(basic_parameters)
        params["sourcedoublecouple"] = "10,10,10"
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200

        assert p.call_count == 1
        assert sorted(p.call_args[1]["components"]) == sorted(
            db.default_components
        )
        assert p.call_args[1][
            "source"
        ] == instaseis.Source.from_strike_dip_rake(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=basic_parameters["sourcedepthinmeters"],
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            strike=10,
            dip=10,
            rake=10,
            M0=1e19,
        )
        assert p.call_args[1]["receiver"] == instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0,
            network="XX",
            station="SYN",
            location="SE",
        )
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
                tr.stats.instaseis = obspy.core.AttribDict()
                tr.stats.instaseis.mu = 1.234
                tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
                tr.stats.delta = dt

            params = copy.deepcopy(basic_parameters)
            params["sourceforce"] = fs_param
            request = fetch_sync(
                client, _assemble_url("seismograms", **params)
            )
            assert request.code == 200

            assert p.call_count == 1
            assert sorted(p.call_args[1]["components"]) == sorted(
                db.default_components
            )
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=0.0,
                origin_time=obspy.UTCDateTime(1900, 1, 1),
                **dict(
                    ("_".join(key), float(value))
                    for (key, value) in fs.items()
                ),
            )
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=0.0,
                network="XX",
                station="SYN",
                location="SE",
            )
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
                tr.stats.instaseis = obspy.core.AttribDict()
                tr.stats.instaseis.mu = 1.234
                tr.stats.starttime = time - 1 - 7 * dt
                tr.stats.delta = dt

            params["sourcedepthinmeters"] = "5.0"
            params["origintime"] = str(time)
            params["receiverdepthinmeters"] = "0.0"
            params["networkcode"] = "BW"
            params["stationcode"] = "ALTM"
            params["locationcode"] = "XX"

            request = fetch_sync(
                client, _assemble_url("seismograms", **params)
            )
            assert request.code == 200

            assert p.call_count == 1
            assert sorted(p.call_args[1]["components"]) == sorted(
                db.default_components
            )
            assert p.call_args[1]["source"] == instaseis.ForceSource(
                latitude=basic_parameters["sourcelatitude"],
                longitude=basic_parameters["sourcelongitude"],
                depth_in_m=5.0,
                origin_time=time,
                **dict(
                    ("_".join(key), float(value))
                    for (key, value) in fs.items()
                ),
            )
            assert p.call_args[1]["receiver"] == instaseis.Receiver(
                latitude=basic_parameters["receiverlatitude"],
                longitude=basic_parameters["receiverlongitude"],
                depth_in_m=0.0,
                network="BW",
                station="ALTM",
                location="XX",
            )
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
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt

        # From here on only 3 component databases.
        if db.info.components != "vertical and horizontal":
            return

        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["components"] = "RTE"
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1) - 7 * dt
            tr.stats.delta = dt
        params = copy.deepcopy(basic_parameters)
        params["sourcemomenttensor"] = mt_param
        params["dt"] = "0.1"
        params["kernelwidth"] = "2"
        params["units"] = "ACCELERATION"
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        "format": "miniseed",
    }

    # Various sources.
    mt = {
        "mrr": "100000",
        "mtt": "200000",
        "mpp": "300000",
        "mrt": "400000",
        "mrp": "500000",
        "mtp": "600000",
    }
    mt_param = "100000,200000,300000,400000,500000,600000"
    sdr = {"strike": "10", "dip": "20", "rake": "30", "M0": "1000000"}
    sdr_param = "10,20,30,1000000"
    fs = {"fr": "100000", "ft": "200000", "fp": "300000"}
    fs_param = "100000,200000,300000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

    components = db.available_components
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0,
        network="XX",
        station="SYN",
        location="SE",
    )
    st_db = db.get_seismograms(
        source=source, receiver=receiver, components=components
    )

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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
        params["receiverdepthinmeters"] = "0.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)
        params["receiverdepthinmeters"] = "55.0"

    params["origintime"] = str(time)
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"
    params["locationcode"] = "XX"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=params["receiverdepthinmeters"],
        network="BW",
        station="ALTM",
        location="XX",
    )
    st_db = db.get_seismograms(
        source=source, receiver=receiver, components=components
    )
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # From strike, dip, rake
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = sdr_param
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict((key, float(value)) for (key, value) in sdr.items()),
    )
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0,
        network="XX",
        station="SYN",
        location="SE",
    )
    st_db = db.get_seismograms(
        source=source, receiver=receiver, components=components
    )
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

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
    params["locationcode"] = "XX"

    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()),
    )
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=params["receiverdepthinmeters"],
        network="BW",
        station="ALTM",
        location="XX",
    )
    st_db = db.get_seismograms(
        source=source, receiver=receiver, components=components
    )
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # Force source only works for displ_only databases.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params["sourceforce"] = fs_param
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        st_server = obspy.read(request.buffer)

        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(
                ("_".join(key), float(value)) for (key, value) in fs.items()
            ),
        )
        receiver = instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=0.0,
            network="XX",
            station="SYN",
            location="SE",
        )
        st_db = db.get_seismograms(
            source=source, receiver=receiver, components=components
        )
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both.
            del tr_server.stats.mseed
            del tr_server.stats._format
            del tr_db.stats.instaseis
            # Sample spacing is very similar but not equal due to floating
            # point accuracy.
            np.testing.assert_allclose(
                tr_server.stats.delta, tr_db.stats.delta
            )
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(
                tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
            )

        params["sourcedepthinmeters"] = "5.0"
        params["receiverdepthinmeters"] = "0.0"

        params["origintime"] = str(time)
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"
        params["locationcode"] = "XX"

        request = fetch_sync(client, _assemble_url("seismograms", **params))
        st_server = obspy.read(request.buffer)

        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=params["sourcedepthinmeters"],
            origin_time=time,
            **dict(
                ("_".join(key), float(value)) for (key, value) in fs.items()
            ),
        )
        receiver = instaseis.Receiver(
            latitude=basic_parameters["receiverlatitude"],
            longitude=basic_parameters["receiverlongitude"],
            depth_in_m=params["receiverdepthinmeters"],
            network="BW",
            station="ALTM",
            location="XX",
        )
        st_db = db.get_seismograms(
            source=source, receiver=receiver, components=components
        )
        for tr_server, tr_db in zip(st_server, st_db):
            # Remove the additional stats from both.
            del tr_server.stats.mseed
            del tr_server.stats._format
            del tr_db.stats.instaseis
            # Sample spacing is very similar but not equal due to floating
            # point accuracy.
            np.testing.assert_allclose(
                tr_server.stats.delta, tr_db.stats.delta
            )
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(
                tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
            )

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0,
        network="XX",
        station="SYN",
        location="SE",
    )

    # Now test other the other parameters.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["components"] = "".join(db.default_components[:1])
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(
        source=source, receiver=receiver, components=db.default_components[:1]
    )
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["units"] = "acceleration"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(
        source=source, receiver=receiver, kind="acceleration"
    )
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["units"] = "velocity"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(
        source=source, receiver=receiver, kind="velocity"
    )
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["dt"] = "0.1"
    params["kernelwidth"] = "1"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(
        source=source, receiver=receiver, dt=0.1, kernelwidth=1
    )
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis

        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["dt"] = "0.1"
    params["kernelwidth"] = "2"
    params["units"] = "ACCELERATION"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)
    st_db = db.get_seismograms(
        source=source,
        receiver=receiver,
        dt=0.1,
        kernelwidth=2,
        kind="acceleration",
        remove_source_shift=True,
    )
    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating
        # point accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )


def test_output_formats(all_clients):
    """
    The /seismograms route can return data either as MiniSEED or as zip
    archive containing multiple SAC files.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }

    # First try to get a MiniSEED file.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # saczip results in a folder of multiple sac files.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        "receiverlongitude": -10,
    }
    mt = {
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "".join(db.default_components[:2]),
        "units": "velocity",
        "dt": 2,
        "kernelwidth": 3,
        "networkcode": "BW",
        "stationcode": "FURT",
        "locationcode": "XX",
    }
    basic_parameters.update(mt)

    # First get a MiniSEED file.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st = obspy.read(request.buffer)
    for tr in st:
        assert tr.stats._format == "MSEED"

    # saczip results in a folder of multiple sac files.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
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
    request = fetch_sync(client, "/coordinates?network=BW&station=FURT")
    assert request.code == 404
    assert request.reason == "Server does not support station coordinates."


def test_coordinates_route_with_stations_coordinates_callback(
    all_clients_station_coordinates_callback,
):
    """
    Tests the /coordinates route.
    """
    client = all_clients_station_coordinates_callback

    # 404 is returned if no coordinates are found.
    request = fetch_sync(client, "/coordinates?network=BW&station=FURT")
    assert request.code == 404
    assert request.reason == "No coordinates found satisfying the query."

    # Single station.
    request = fetch_sync(client, "/coordinates?network=IU&station=ANMO")
    assert request.code == 200
    # Assert the GeoJSON content-type.
    assert request.headers["Content-Type"] == "application/vnd.geo+json"
    stations = json.loads(str(request.body.decode("utf8")))

    assert stations == {
        "features": [
            {
                "geometry": {
                    "coordinates": [-106.4572, 34.94591],
                    "type": "Point",
                },
                "properties": {"network_code": "IU", "station_code": "ANMO"},
                "type": "Feature",
            }
        ],
        "type": "FeatureCollection",
    }

    # Multiple stations with wildcard searches.
    request = fetch_sync(
        client, "/coordinates?network=IU,B*&station=ANT*,ANM?"
    )
    assert request.code == 200
    # Assert the GeoJSON content-type.
    assert request.headers["Content-Type"] == "application/vnd.geo+json"
    stations = json.loads(str(request.body.decode("utf8")))

    assert stations == {
        "features": [
            {
                "geometry": {
                    "coordinates": [32.7934, 39.868],
                    "type": "Point",
                },
                "properties": {"network_code": "IU", "station_code": "ANTO"},
                "type": "Feature",
            },
            {
                "geometry": {
                    "coordinates": [-106.4572, 34.94591],
                    "type": "Point",
                },
                "properties": {"network_code": "IU", "station_code": "ANMO"},
                "type": "Feature",
            },
        ],
        "type": "FeatureCollection",
    }

    # network and station must be given.
    request = fetch_sync(client, "/coordinates?network=IU")
    assert request.code == 400
    assert request.reason == (
        "Parameters 'network' and 'station' must be " "given."
    )


def test_cors_headers(all_clients_all_callbacks):
    """
    Check that all routes return CORS headers.
    """
    client = all_clients_all_callbacks

    request = fetch_sync(client, "/")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = fetch_sync(client, "/info")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = fetch_sync(client, "/coordinates?network=IU&station=ANMO")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = fetch_sync(client, "/event?id=B071791B")
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=%i&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phases=P" % client.source_depth,
    )
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
        "mtp": "100000",
    }
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
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
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"


def test_cors_headers_failing_requests(all_clients_all_callbacks):
    """
    Check that all routes return CORS headers also for failing requests.
    """
    client = all_clients_all_callbacks

    request = fetch_sync(client, "/coordinates")
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = fetch_sync(client, "/event")
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    request = fetch_sync(client, "/ttimes")
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    # raw seismograms route
    params = {}
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"

    # standard seismograms route
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert "Access-Control-Allow-Origin" in request.headers
    assert request.headers["Access-Control-Allow-Origin"] == "*"


def test_gzipped_responses(all_clients_all_callbacks):
    """
    The JSON responses should all be gzipped if requested.

    Starting with tornado 4.3 responses smaller than 1000 bytes do no longer
    get compressed. Thus we can also test some responses here.
    """
    client = all_clients_all_callbacks

    # Explicitly turn if off.
    request = fetch_sync(client, "/info", use_gzip=False)
    assert request.code == 200
    assert "X-Consumed-Content-Encoding" not in request.headers
    request = fetch_sync(client, "/info", use_gzip=True)
    assert request.code == 200
    assert request.headers["X-Consumed-Content-Encoding"] == "gzip"

    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=%i&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phases=P" % client.source_depth,
    )
    assert request.code == 200
    assert "X-Consumed-Content-Encoding" not in request.headers

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
        "mtp": "100000",
    }
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 200
    assert "X-Consumed-Content-Encoding" not in request.headers
    # Gzipping should not return gzipped data as its a binary format.
    request = fetch_sync(
        client, _assemble_url("seismograms_raw", **params), use_gzip=True
    )
    assert request.code == 200
    assert "X-Consumed-Content-Encoding" not in request.headers

    # standard seismograms route
    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    assert "X-Consumed-Content-Encoding" not in request.headers
    # Binary format, no gzipping.
    request = fetch_sync(
        client, _assemble_url("seismograms", **params), use_gzip=True
    )
    assert request.code == 200
    assert "X-Consumed-Content-Encoding" not in request.headers


def test_multiple_seismograms_retrieval_no_format_given(
    all_clients_station_coordinates_callback,
):
    """
    Tests  the retrieval of multiple station in one request with no passed
    format parameter. This results in saczip return values.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
    }

    # Various sources.
    mt = {
        "mtt": "100000",
        "mpp": "100000",
        "mrr": "100000",
        "mrt": "100000",
        "mrp": "100000",
        "mtp": "100000",
    }
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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    st_server.sort()

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receivers = [
        instaseis.Receiver(
            latitude=39.868,
            longitude=32.7934,
            depth_in_m=0.0,
            network="IU",
            station="ANTO",
        ),
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        ),
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver)
    st_db.sort()

    assert len(st_db) == len(db.default_components) * 2
    assert len(st_server) == len(db.default_components) * 2
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # Strike/dip/rake source
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = sdr_param
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"
    # A couple more parameters.
    if client.is_reciprocal is True:
        params["sourcedepthinmeters"] = "5.0"
    params["origintime"] = str(time)

    # Default format is MiniSEED>
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    st_server.sort()

    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=5.0,
        origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()),
    )
    receivers = [
        instaseis.Receiver(
            latitude=39.868,
            longitude=32.7934,
            depth_in_m=0.0,
            network="IU",
            station="ANTO",
        ),
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        ),
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver)
    st_db.sort()

    # Should now have only 4 Stream objects.
    assert len(st_db) == len(db.default_components) * 2
    assert len(st_server) == len(db.default_components) * 2
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # Force source only works for displ_only databases.
    # Force source, all components.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params["sourceforce"] = fs_param
        # This will return two stations.
        params["network"] = "IU,B*"
        params["station"] = "ANT*,ANM?"
        params["components"] = "".join(db.available_components)

        # Default format is MiniSEED>
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200
        assert request.headers["Content-Type"] == "application/zip"
        st_server = obspy.Stream()
        zip_obj = zipfile.ZipFile(request.buffer)
        for name in zip_obj.namelist():
            st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
        st_server.sort()

        components = db.available_components
        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(
                ("_".join(key), float(value)) for (key, value) in fs.items()
            ),
        )
        receivers = [
            instaseis.Receiver(
                latitude=39.868,
                longitude=32.7934,
                depth_in_m=0.0,
                network="IU",
                station="ANTO",
            ),
            instaseis.Receiver(
                latitude=34.94591,
                longitude=-106.4572,
                depth_in_m=0.0,
                network="IU",
                station="ANMO",
            ),
        ]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(
                source=source, receiver=receiver, components=components
            )
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
            np.testing.assert_allclose(
                tr_server.stats.delta, tr_db.stats.delta
            )
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(
                tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
            )


def test_multiple_seismograms_retrieval_no_format_given_single_station(
    all_clients_station_coordinates_callback,
):
    """
    Tests  the retrieval of multiple station in one request with no passed
    format parameter. This results in sac return values.

    In this case the query is constructed so that it only returns a single
    station.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
    }

    # Various sources.
    mt = {
        "mrr": "100000",
        "mtt": "200000",
        "mpp": "300000",
        "mrt": "400000",
        "mrp": "500000",
        "mtp": "600000",
    }
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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/zip"
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    st_server.sort()

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receivers = [
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        )
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver)
    st_db.sort()

    assert len(st_db) == len(db.default_components)
    assert len(st_server) == len(db.default_components)
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # From this point on, only three component databases.
    if "R" not in db.available_components:
        return

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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        **dict((key, float(value)) for (key, value) in sdr.items()),
    )
    receivers = [
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        )
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(
            source=source, receiver=receiver, components=components
        )
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

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
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
            depth_in_m=0.0,
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(
                ("_".join(key), float(value)) for (key, value) in fs.items()
            ),
        )
        receivers = [
            instaseis.Receiver(
                latitude=34.94591,
                longitude=-106.4572,
                depth_in_m=0.0,
                network="IU",
                station="ANMO",
            )
        ]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(
                source=source, receiver=receiver, components=components
            )
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
            np.testing.assert_allclose(
                tr_server.stats.delta, tr_db.stats.delta
            )
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(
                tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
            )


def test_multiple_seismograms_retrieval_mseed_format(
    all_clients_station_coordinates_callback,
):
    """
    Tests  the retrieval of multiple station in one request with the mseed
    format parameter.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "format": "miniseed",
        "sourcedepthinmeters": client.source_depth,
    }

    # Various sources.
    mt = {
        "mtt": "100000",
        "mpp": "100000",
        "mrr": "100000",
        "mrt": "100000",
        "mrp": "100000",
        "mtp": "100000",
    }
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

    # Default format is MiniSEED.
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/vnd.fdsn.mseed"
    st_server = obspy.read(request.buffer)
    st_server.sort()

    components = db.default_components
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receivers = [
        instaseis.Receiver(
            latitude=39.868,
            longitude=32.7934,
            depth_in_m=0.0,
            network="IU",
            station="ANTO",
        ),
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        ),
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(
            source=source, receiver=receiver, components=components
        )
    st_db.sort()

    # Should now have a number of streams.

    assert len(st_db) > 1
    assert len(st_db) == len(st_server)

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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # Execute the rest only for databases that have vertical and horizontal
    # components.
    if sorted(db.default_components) != ["E", "N", "Z"]:
        return

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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    assert request.headers["Content-Type"] == "application/vnd.fdsn.mseed"
    st_server = obspy.read(request.buffer)
    st_server.sort()

    components = ["R", "T"]
    source = instaseis.Source.from_strike_dip_rake(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=params["sourcedepthinmeters"],
        origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()),
    )
    receivers = [
        instaseis.Receiver(
            latitude=39.868,
            longitude=32.7934,
            depth_in_m=0.0,
            network="IU",
            station="ANTO",
        ),
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        ),
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(
            source=source, receiver=receiver, components=components
        )
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

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
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200
        assert request.headers["Content-Type"] == "application/vnd.fdsn.mseed"
        st_server = obspy.read(request.buffer)
        st_server.sort()

        components = ["N", "E", "Z", "R", "T"]
        source = instaseis.ForceSource(
            latitude=basic_parameters["sourcelatitude"],
            longitude=basic_parameters["sourcelongitude"],
            depth_in_m=0.0,
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(
                ("_".join(key), float(value)) for (key, value) in fs.items()
            ),
        )
        receivers = [
            instaseis.Receiver(
                latitude=39.868,
                longitude=32.7934,
                depth_in_m=0.0,
                network="IU",
                station="ANTO",
            ),
            instaseis.Receiver(
                latitude=34.94591,
                longitude=-106.4572,
                depth_in_m=0.0,
                network="IU",
                station="ANMO",
            ),
        ]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(
                source=source, receiver=receiver, components=components
            )
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
            np.testing.assert_allclose(
                tr_server.stats.delta, tr_db.stats.delta
            )
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(
                tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
            )


def test_multiple_seismograms_retrieval_saczip_format(
    all_clients_station_coordinates_callback,
):
    """
    Tests  the retrieval of multiple station in one request with the saczip
    format parameter.
    """
    client = all_clients_station_coordinates_callback
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "format": "saczip",
    }

    # Various sources.
    mt = {
        "mtt": "100000",
        "mpp": "100000",
        "mrr": "100000",
        "mrt": "100000",
        "mrp": "100000",
        "mtp": "100000",
    }
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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
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

    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receivers = [
        instaseis.Receiver(
            latitude=39.868,
            longitude=32.7934,
            depth_in_m=0.0,
            network="IU",
            station="ANTO",
        ),
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        ),
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(source=source, receiver=receiver)
    st_db.sort()

    # Should now have the number of default components times two (once for
    # each station)
    assert len(st_db) == len(db.default_components) * 2
    assert len(st_server) == len(db.default_components) * 2
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # From here on only 3C databases
    if "R" not in db.available_components:
        return

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
    request = fetch_sync(client, _assemble_url("seismograms", **params))
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
        depth_in_m=5.0,
        origin_time=time,
        **dict((key, float(value)) for (key, value) in sdr.items()),
    )
    receivers = [
        instaseis.Receiver(
            latitude=39.868,
            longitude=32.7934,
            depth_in_m=0.0,
            network="IU",
            station="ANTO",
        ),
        instaseis.Receiver(
            latitude=34.94591,
            longitude=-106.4572,
            depth_in_m=0.0,
            network="IU",
            station="ANMO",
        ),
    ]
    st_db = obspy.Stream()
    for receiver in receivers:
        st_db += db.get_seismograms(
            source=source, receiver=receiver, components=components
        )
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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

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
        request = fetch_sync(client, _assemble_url("seismograms", **params))
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
            depth_in_m=0.0,
            origin_time=obspy.UTCDateTime(1900, 1, 1),
            **dict(
                ("_".join(key), float(value)) for (key, value) in fs.items()
            ),
        )
        receivers = [
            instaseis.Receiver(
                latitude=39.868,
                longitude=32.7934,
                depth_in_m=0.0,
                network="IU",
                station="ANTO",
            ),
            instaseis.Receiver(
                latitude=34.94591,
                longitude=-106.4572,
                depth_in_m=0.0,
                network="IU",
                station="ANMO",
            ),
        ]
        st_db = obspy.Stream()
        for receiver in receivers:
            st_db += db.get_seismograms(
                source=source, receiver=receiver, components=components
            )
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
            np.testing.assert_allclose(
                tr_server.stats.delta, tr_db.stats.delta
            )
            tr_server.stats.delta = tr_db.stats.delta
            assert tr_server.stats == tr_db.stats
            # Relative tolerance not particularly useful when testing super
            # small values.
            np.testing.assert_allclose(
                tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
            )


def test_multiple_seismograms_retrieval_invalid_format(
    all_clients_station_coordinates_callback,
):
    """
    Tests  the retrieval of multiple station with an invalid format.
    """
    client = all_clients_station_coordinates_callback

    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "format": "bogus",
    }
    # This will return two stations.
    params["network"] = "IU,B*"
    params["station"] = "ANT*,ANM?"

    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == "Format must either be 'miniseed' or 'saczip'."


def test_multiple_seismograms_retrieval_no_stations(
    all_clients_station_coordinates_callback,
):
    """
    Tests  the retrieval of multiple station where the request ends up in no
    found stations.
    """
    client = all_clients_station_coordinates_callback

    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }
    # This will return two stations.
    params["network"] = "HE"
    params["station"] = "LLO"

    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 404
    assert request.reason == "No coordinates found satisfying the query."


def test_unknown_parameter_raises(all_clients):
    """
    Unknown parameters should raise.
    """
    client = all_clients

    # Normal request works fine.
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
        "mtp": "100000",
    }
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 200

    # Adding a random other parameter raises
    params["bogus"] = "bogus"
    params["random"] = "stuff"
    request = fetch_sync(client, _assemble_url("seismograms_raw", **params))
    assert request.code == 400
    assert request.reason == (
        "The following unknown parameters have been "
        "passed: 'bogus', 'random'"
    )

    # Same with /seismograms route.
    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200

    # Adding a random other parameter raises
    params["random"] = "stuff"
    params["bogus"] = "bogus"
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "The following unknown parameters have been "
        "passed: 'bogus', 'random'"
    )


def test_passing_duplicate_parameter_raises(all_clients):
    """
    While valid with HTTP, duplicate parameters are not allowed within
    instaseis. This should thus raise an error to avoid confusion of users.
    """
    client = all_clients

    # Normal request works fine.
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
        "mtp": "100000",
    }
    url = _assemble_url("seismograms_raw", **params)
    request = fetch_sync(client, url)
    assert request.code == 200

    # Adding two duplicate parameters raises.
    url += "&receiverlatitude=10&mrt=10"
    request = fetch_sync(client, url)
    assert request.code == 400
    assert request.reason == (
        "Duplicate parameters: 'mrt', " "'receiverlatitude'"
    )

    # Same with /seismograms route.
    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }
    url = _assemble_url("seismograms", **params)
    request = fetch_sync(client, url)
    assert request.code == 200

    # Adding two duplicate parameters raises.
    url += "&receiverlatitude=10&sourcemomenttensor=10"
    request = fetch_sync(client, url)
    assert request.code == 400
    assert request.reason == (
        "Duplicate parameters: 'receiverlatitude', 'sourcemomenttensor'"
    )


def test_passing_invalid_time_settings_raises(all_clients):
    """
    Tests that invalid time settings raise.
    """
    origin_time = obspy.UTCDateTime(2015, 1, 1)
    client = all_clients
    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "origintime": str(origin_time),
    }

    # This should work fine.
    url = _assemble_url("seismograms", **params)
    request = fetch_sync(client, url)
    assert request.code == 200

    # The remainder should not.
    p = copy.deepcopy(params)
    p["starttime"] = str(origin_time + 1e6)
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    assert request.code == 400
    assert request.reason == (
        "The `starttime` must be before the seismogram " "ends."
    )

    p = copy.deepcopy(params)
    p["endtime"] = str(origin_time - 1e6)
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    assert request.code == 400
    assert request.reason == (
        "The end time of the seismograms lies outside " "the allowed range."
    )

    p = copy.deepcopy(params)
    p["starttime"] = str(origin_time - 3800)
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    assert request.code == 400
    assert request.reason == (
        "The seismogram can start at the maximum one "
        "hour before the origin time."
    )


def test_time_settings_for_seismograms_route(all_clients):
    """
    Tests the advanced time settings.
    """
    client = all_clients

    origin_time = obspy.UTCDateTime(2015, 1, 1)

    client = all_clients
    # Resample to 1Hz to simplify the logic.
    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 1.0,
        "kernelwidth": 1,
        "origintime": str(origin_time),
        "format": "miniseed",
    }

    p = copy.deepcopy(params)
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    st = obspy.read(request.buffer)

    # This should start at the origin time.
    for tr in st:
        assert tr.stats.starttime == origin_time

    # Different starttime.
    p = copy.deepcopy(params)
    p["starttime"] = origin_time - 10
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time - 10

    # Can also be given as a float in which case it will be interpreted as
    # an offset to the origin time.
    p = copy.deepcopy(params)
    p["starttime"] = -10
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
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
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time
        assert tr.stats.endtime == origin_time + 10

    # Can also be done by passing a float which will be interpreted as an
    # offset in respect to the starttime.
    p = copy.deepcopy(params)
    p["endtime"] = 13.0
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time
        assert tr.stats.endtime == origin_time + 13

    # If starttime is given, the duration is relative to the starttime.
    p = copy.deepcopy(params)
    p["endtime"] = 10
    p["starttime"] = origin_time - 5
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
    st_2 = obspy.read(request.buffer)

    for tr in st_2:
        assert tr.stats.starttime == origin_time - 5
        assert tr.stats.endtime == origin_time + 5

    # Will be padded with zeros in the front. Attempting to pad with zeros
    # in the back will raise an error but that is tested elsewhere.
    p = copy.deepcopy(params)
    p["starttime"] = origin_time - 1800
    url = _assemble_url("seismograms", **p)
    request = fetch_sync(client, url)
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
    request = fetch_sync(client, "/event?id=B071791B")
    assert request.code == 404
    assert request.reason == "Server does not support event information."


def test_event_route_with_event_coordinates_callback(
    all_clients_event_callback,
):
    """
    Tests the /event route.
    """
    client = all_clients_event_callback

    # Missing 'id' parameter.
    request = fetch_sync(client, "/event")
    assert request.code == 400
    assert request.reason == "'id' parameter is required."

    # Unknown event.
    request = fetch_sync(client, "/event?id=bogus")
    assert request.code == 404
    assert request.reason == "Event not found."

    # Known event.
    request = fetch_sync(client, "/event?id=B071791B")
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
        "origin_time": "1991-07-17T16:41:33.100000Z",
    }


def test_station_query_various_failures(
    all_clients_station_coordinates_callback,
):
    """
    The station query can fail for various reasons.
    """
    client = all_clients_station_coordinates_callback

    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }

    # It fails if receiver coordinates and query params are passed.
    p = copy.deepcopy(params)
    p["network"] = "IU,B*"
    p["station"] = "ANT*,ANM?"
    p["receiverlatitude"] = 1.0

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == (
        "Receiver coordinates can either be specified by passing "
        "the coordinates, or by specifying query parameters, "
        "but not both."
    )

    # It also fails if only one part is given.
    p = copy.deepcopy(params)
    p["network"] = "IU,B*"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == (
        "Must specify a full set of coordinates or a full set of receiver "
        "parameters."
    )

    # It also fails if it does not find any networks and stations.
    p = copy.deepcopy(params)
    p["network"] = "X*"
    p["station"] = "Y*"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 404
    assert request.reason == ("No coordinates found satisfying the query.")

    # Trigger a very specific error occuring when the station coordinate
    # callback yields invalid coordinates.
    p = copy.deepcopy(params)
    p["network"] = "XX"
    p["station"] = "DUMMY"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == (
        "Could not construct receiver with passed "
        "parameters. Check parameters for sanity."
    )


def test_station_query_no_callback(all_clients):
    """
    Test the error message when no station callback is available.
    """
    client = all_clients

    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": 0,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
    }

    p = copy.deepcopy(params)
    p["network"] = "IU,B*"
    p["station"] = "ANT*,ANM?"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support station coordinates and thus no station "
        "queries."
    )


def test_event_query_no_callbacks(all_clients):
    """
    Test the error message when no event callback is available.
    """
    client = all_clients

    params = {"receiverlatitude": 10, "receiverlongitude": 10}

    p = copy.deepcopy(params)
    p["eventid"] = "B071791B"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support event information and thus no event queries."
    )


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
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == (
        "The following parameters cannot be used if 'eventid' is a "
        "parameter: 'sourcedepthinmeters', 'sourcelatitude', "
        "'sourcelongitude', 'sourcemomenttensor'"
    )

    # Neither can the origin time be specified.
    p = copy.deepcopy(params)
    p["eventid"] = "B071791B"
    p["origintime"] = obspy.UTCDateTime(2014, 1, 1)

    # Cannot not pass other source parameters along.
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == (
        "'eventid' and 'origintime' parameters cannot both be passed at the "
        "same time."
    )

    # Callback returns an invalid event.
    p = copy.deepcopy(params)
    p["eventid"] = "invalid_event"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == ("Event callback returned an invalid result.")


def test_event_parameters_by_querying(all_clients_event_callback):
    """
    Test the query by eventid.
    """
    client = all_clients_event_callback

    # Only works for reciprocal databases. Otherwise the depth if fixed.
    if not client.is_reciprocal:
        return

    db = instaseis.open_db(client.filepath)

    params = {
        "receiverlatitude": 10,
        "receiverlongitude": 10,
        "format": "miniseed",
    }

    p = copy.deepcopy(params)
    p["eventid"] = "B071791B"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
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
        origin_time=obspy.UTCDateTime("1991-07-17T16:41:33.100000Z"),
    )
    receiver = instaseis.Receiver(
        latitude=10,
        longitude=10,
        depth_in_m=0.0,
        network="XX",
        station="SYN",
        location="SE",
    )

    st_db = db.get_seismograms(source=source, receiver=receiver)

    for tr, tr_db in zip(st, st_db):
        del tr.stats._format
        del tr.stats.mseed
        del tr_db.stats.instaseis

        np.testing.assert_allclose([tr.stats.delta], [tr_db.stats.delta])
        tr.stats.delta = tr_db.stats.delta
        assert tr.stats == tr_db.stats
        np.testing.assert_allclose(
            tr.data, tr_db.data, atol=tr.data.ptp() / 1e9
        )

    # Also perform a mock comparison to test the actually created object.
    with mock.patch(
        "instaseis.database_interfaces.base_netcdf_instaseis_db"
        ".BaseNetCDFInstaseisDB.get_seismograms"
    ) as patch:
        _st = obspy.read()
        for tr in _st:
            tr.stats.instaseis = obspy.core.AttribDict()
            tr.stats.instaseis.mu = 1.234
            tr.stats.starttime = source.origin_time - 0.01
            tr.stats.delta = 10.0
        patch.return_value = _st
        result = fetch_sync(client, _assemble_url("seismograms", **p))
        assert result.code == 200

    assert patch.call_args[1]["source"] == source


def test_event_query_seismogram_non_existent_event(all_clients_event_callback):
    """
    Tests querying for an event that is not found.
    """
    client = all_clients_event_callback

    params = {
        "receiverlatitude": 10,
        "receiverlongitude": 10,
        "eventid": "bogus",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 404
    assert request.reason == "Event not found."


def test_mu_parameter_for_seismograms_and_greens_function_route(
    all_clients_station_coordinates_callback,
):
    """
    Test that the mu parameter is passed on for seismograms anf the greens
    function route.
    """
    client = all_clients_station_coordinates_callback

    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "receiverlatitude": 20,
        "receiverlongitude": 20,
        "format": "miniseed",
    }

    p = copy.deepcopy(params)
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200

    # Make sure the mu header exists and the value can be converted to a float.
    assert "Instaseis-Mu" in request.headers
    assert isinstance(float(request.headers["Instaseis-Mu"]), float)

    # Multiple stations. mu is source dependent and thus the same.
    p = copy.deepcopy(params)
    del p["receiverlatitude"]
    del p["receiverlongitude"]
    # This will return two stations.
    p["network"] = "IU,B*"
    p["station"] = "ANT*,ANM?"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200

    # Make sure the mu header exists and the value can be converted to a float.
    assert "Instaseis-Mu" in request.headers
    assert isinstance(float(request.headers["Instaseis-Mu"]), float)

    # get_greens_function() only works with reciprocal DBs that also must
    # have all three components.
    if (
        not client.is_reciprocal
        or client.info.components != "vertical and horizontal"
    ):
        return

    parameters = {
        "sourcedepthinmeters": 1e3,
        "sourcedistanceindegrees": 20,
        "format": "saczip",
    }

    # default parameters
    request = fetch_sync(
        client, _assemble_url("greens_function", **parameters)
    )
    assert request.code == 200

    # Make sure the mu header exists and the value can be converted to a float.
    assert "Instaseis-Mu" in request.headers
    assert isinstance(float(request.headers["Instaseis-Mu"]), float)


def test_label_parameter(all_greens_clients):
    """
    Test the 'label' parameter of the /seismograms route.
    """
    prefix = "attachment; filename="
    client = all_greens_clients

    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "receiverlatitude": 20,
        "receiverlongitude": 20,
        "format": "miniseed",
    }

    # No specified label will result in it having a generic label.
    p = copy.deepcopy(params)

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix) :]  # NOQA
    assert filename.startswith("instaseis_seismogram_")
    assert filename.endswith(".mseed")

    # The same is true if saczip is used but in that case the ending is zip
    # and all the files inside have the trace id as the name.
    p = copy.deepcopy(params)
    p["format"] = "saczip"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix) :]  # NOQA
    assert filename.startswith("instaseis_seismogram_")
    assert filename.endswith(".zip")

    zip_obj = zipfile.ZipFile(request.buffer)
    names = zip_obj.namelist()
    zip_obj.close()

    assert sorted(names) == sorted(
        ["XX.SYN.SE.LXZ.sac", "XX.SYN.SE.LXN.sac", "XX.SYN.SE.LXE.sac"]
    )

    # Now pass one. It will replace the filename prefix.
    p = copy.deepcopy(params)
    p["label"] = "Tohoku"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix) :]  # NOQA
    assert filename.startswith("Tohoku_")
    assert filename.endswith(".mseed")

    # Same for saczip and also the files in the zip should change.
    p = copy.deepcopy(params)
    p["format"] = "saczip"
    p["label"] = "Tohoku"

    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200

    filename = request.headers["Content-Disposition"]
    assert filename.startswith(prefix)

    filename = filename[len(prefix) :]  # NOQA
    assert filename.startswith("Tohoku_")
    assert filename.endswith(".zip")

    zip_obj = zipfile.ZipFile(request.buffer)
    names = zip_obj.namelist()
    zip_obj.close()

    assert sorted(names) == sorted(
        [
            "Tohoku_XX.SYN.SE.LXZ.sac",
            "Tohoku_XX.SYN.SE.LXN.sac",
            "Tohoku_XX.SYN.SE.LXE.sac",
        ]
    )


def test_ttimes_route_no_callback(all_clients):
    """
    Tests the ttimes route with no available callbacks.
    """
    client = all_clients

    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phase=P",
    )
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations."
    )


def test_ttimes_route(all_clients_ttimes_callback):
    """
    Test for the ttimes route.
    """
    client = all_clients_ttimes_callback

    # Test with missing parameters.
    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90",
    )
    assert request.code == 400
    assert request.reason == (
        "The following required parameters are missing: "
        "'phases', 'receiverdepthinmeters'"
    )

    # Invalid phase name
    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phases=bogs",
    )
    assert request.code == 400
    assert request.reason == "Invalid phase name 'bogs'."

    # Other error, e.g. negative depth.
    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=-200&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phases=P",
    )
    assert request.code == 400
    assert request.reason == (
        "Failed to calculate travel time due to: No "
        "layer contains this depth"
    )

    # No such phase at that distance.
    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=0&phases=Pdiff",
    )
    assert request.code == 404
    assert (
        request.reason
        == "No ray for the given geometry and any of the phases found."
    )

    # Many implementations will not have a receiverdepth. This one does not.
    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=50&sourcelongitude=10&"
        "sourcedepthinmeters=0&receiverlatitude=40&receiverlongitude=90&"
        "receiverdepthinmeters=20&phases=Pdiff",
    )
    assert request.code == 400
    assert request.reason == (
        "Failed to calculate travel time due to: This "
        "travel time implementation cannot calculate "
        "buried receivers."
    )

    # Last but not least test some actual travel times.
    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=0&sourcelongitude=0&"
        "sourcedepthinmeters=300000&receiverlatitude=0&receiverlongitude=50&"
        "receiverdepthinmeters=0&phases=P",
    )
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert list(result.keys()) == ["travel_times"]
    assert abs(result["travel_times"]["P"] - 504.357 < 1e-2)

    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=0&sourcelongitude=0&"
        "sourcedepthinmeters=300000&receiverlatitude=0&receiverlongitude=50&"
        "receiverdepthinmeters=0&phases=PP",
    )
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert list(result.keys()) == ["travel_times"]
    assert abs(result["travel_times"]["PP"] - 622.559 < 1e-2)

    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=0&sourcelongitude=0&"
        "sourcedepthinmeters=300000&receiverlatitude=0&receiverlongitude=50&"
        "receiverdepthinmeters=0&phases=sPKiKP",
    )
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert list(result.keys()) == ["travel_times"]
    assert abs(result["travel_times"]["sPKiKP"] - 1090.081 < 1e-2)

    # Multiple phases at once with one phase not existing at hte distance.
    request = fetch_sync(
        client,
        "/ttimes?sourcelatitude=0&sourcelongitude=0&"
        "sourcedepthinmeters=300000&receiverlatitude=0&receiverlongitude=50&"
        "receiverdepthinmeters=0&phases=sPKiKP,P,PP,Sdiff",
    )
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert list(result.keys()) == ["travel_times"]
    assert sorted(list(result["travel_times"].keys())) == ["P", "PP", "sPKiKP"]
    assert abs(result["travel_times"]["P"] - 504.357 < 1e-2)
    assert abs(result["travel_times"]["PP"] - 622.559 < 1e-2)
    assert abs(result["travel_times"]["sPKiKP"] - 1090.081 < 1e-2)


def test_network_and_station_code_settings(all_clients):
    """
    Tests the network and station code settings.
    """
    client = all_clients

    params = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "receiverlatitude": 20,
        "receiverlongitude": 20,
        "format": "miniseed",
    }

    # Default network and station codes.
    p = copy.deepcopy(params)
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "XX"
        assert tr.stats.station == "SYN"

    # Set only the network code.
    p = copy.deepcopy(params)
    p["networkcode"] = "BW"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "BW"
        assert tr.stats.station == "SYN"
        assert tr.stats.location == "SE"

    # Set only the station code.
    p = copy.deepcopy(params)
    p["stationcode"] = "INS"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "XX"
        assert tr.stats.station == "INS"
        assert tr.stats.location == "SE"

    # Set both.
    p = copy.deepcopy(params)
    p["networkcode"] = "BW"
    p["stationcode"] = "INS"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "BW"
        assert tr.stats.station == "INS"
        assert tr.stats.location == "SE"

    # Set only the location code.
    p = copy.deepcopy(params)
    p["locationcode"] = "AA"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    for tr in obspy.read(request.buffer):
        assert tr.stats.network == "XX"
        assert tr.stats.station == "SYN"
        assert tr.stats.location == "AA"

    # Station code is limited to five letters.
    p = copy.deepcopy(params)
    p["stationcode"] = "123456"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == "'stationcode' must have 5 or fewer letters."

    # Network code is limited to two letters.
    p = copy.deepcopy(params)
    p["networkcode"] = "123"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == "'networkcode' must have 2 or fewer letters."

    # Location code is limited to two letters.
    p = copy.deepcopy(params)
    p["locationcode"] = "123"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == "'locationcode' must have 2 or fewer letters."


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
        "sourcelatitude": 0,
        "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0,
        "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 0.1,
        "format": "miniseed",
    }

    # Normal seismogram.
    p = copy.deepcopy(params)
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    starttime, endtime = tr.stats.starttime, tr.stats.endtime

    # Start 10 seconds before the P arrival.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Starts 10 seconds after the P arrival
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # Ends 15 seconds before the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2D15"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 15)) < 0.1

    # Ends 15 seconds after the PP arrival
    p = copy.deepcopy(params)
    p["endtime"] = "PP%2B15"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1

    # Starts 5 seconds before the PP and ends 2 seconds after the sPKiKP phase.
    p = copy.deepcopy(params)
    p["starttime"] = "PP%2D5"
    p["endtime"] = "sPKiKP%2B2"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
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
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert abs((tr.stats.starttime) - (starttime + 622.559 - 5)) < 0.1
    assert abs((tr.stats.endtime) - (starttime + 622.559 - 5 + 10)) < 0.1

    # Nonetheless, also test the other combination of relative start time
    # and phase relative endtime.
    p = copy.deepcopy(params)
    p["starttime"] = "10"
    p["endtime"] = "PP%2B15"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]

    assert tr.stats.starttime == starttime + 10
    assert abs((tr.stats.endtime) - (starttime + 622.559 + 15)) < 0.1


def test_phase_relative_offsets_but_no_ttimes_callback(all_clients):
    client = all_clients

    params = {
        "sourcelatitude": 0,
        "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0,
        "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z",
        "dt": 0.1,
        "format": "miniseed",
    }

    # Test for starttime.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations."
    )

    # Test for endtime.
    p = copy.deepcopy(params)
    p["endtime"] = "P%2D10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations."
    )

    # Test for both.
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    p["endtime"] = "S%2B10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 404
    assert request.reason == (
        "Server does not support travel time calculations."
    )


def test_phase_relative_offset_different_time_representations(
    all_clients_ttimes_callback,
):
    client = all_clients_ttimes_callback

    # Different source depth for non-reciprocal client...
    if not client.is_reciprocal:
        return

    params = {
        "sourcelatitude": 0,
        "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0,
        "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 0.1,
        "format": "miniseed",
    }

    # Normal seismogram.
    p = copy.deepcopy(params)
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    starttime, endtime = tr.stats.starttime, tr.stats.endtime

    # P+10
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-10
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+10.0
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10.0"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-10.0
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10.0"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+10.000
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D10.000"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-10.000
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B10.000"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+1E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D1E1"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-1E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B1E1"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+1.0E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D1.0E1"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-1.0E1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B1.0E1"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 + 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P+1e1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2D1e1"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 200
    tr = obspy.read(request.buffer)[0]
    assert abs((tr.stats.starttime) - (starttime + 504.357 - 10)) < 0.1
    assert tr.stats.endtime == endtime

    # P-1e1
    p = copy.deepcopy(params)
    p["starttime"] = "P%2B1e1"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
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
        "sourcelatitude": 0,
        "sourcelongitude": 0,
        "sourcedepthinmeters": 300000,
        "receiverlatitude": 0,
        "receiverlongitude": 50,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "components": "Z",
        "dt": 0.1,
        "format": "miniseed",
    }

    # Illegal phase.
    p = copy.deepcopy(params)
    p["starttime"] = "bogus%2D10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == "Invalid phase name: bogus"

    # Phase not available at that distance.
    p = copy.deepcopy(params)
    p["starttime"] = "Pdiff%2D10"
    request = fetch_sync(client, _assemble_url("seismograms", **p))
    assert request.code == 400
    assert request.reason == (
        "No seismograms found for the given phase relative "
        "offsets. This could either be due to the chosen phase "
        "not existing for the specific source-receiver geometry "
        "or arriving too late/with too large offsets if the "
        "database is not long enough."
    )


def test_phase_relative_offsets_multiple_stations(all_clients_all_callbacks):
    client = all_clients_all_callbacks

    # Only for reciprocal ones as the depth is different otherwise...
    if not client.is_reciprocal:
        return

    db = instaseis.open_db(client.filepath, read_on_demand=True)

    # Now test multiple receiveers.
    # This is constructed in such a way that only one station will have a P
    # phase (due to the distance). So this is tested here.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
        "starttime": "P%2D10",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert len(st) == len(db.default_components)

    # This is constructed in such a way that only one station will have a P
    # phase (due to the distance). So this is tested here.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
        "endtime": "P%2D10",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    st = obspy.read(request.buffer)
    assert len(st) == len(db.default_components)

    # Now get both.
    params = {
        "sourcelatitude": 39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
        "starttime": "P%2D10",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    st = obspy.read(request.buffer)
    # Two stations!
    assert len(st) == len(db.default_components) * 2

    # Or one also does not get any. In that case an error is raised.
    params = {
        "sourcelatitude": 39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
        "starttime": "P%2D10000",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "No seismograms found for the given phase relative "
        "offsets. This could either be due to the chosen phase "
        "not existing for the specific source-receiver geometry "
        "or arriving too late/with too large offsets if the "
        "database is not long enough."
    )

    # Or one also does not get any. In that case an error is raised.
    params = {
        "sourcelatitude": 39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,100000",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
        "endtime": "P%2B10000",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "No seismograms found for the given phase relative "
        "offsets. This could either be due to the chosen phase "
        "not existing for the specific source-receiver geometry "
        "or arriving too late/with too large offsets if the "
        "database is not long enough."
    )


def test_various_failure_conditions(all_clients_all_callbacks):
    client = all_clients_all_callbacks
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    # no source mechanism given.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "One of the following has to be given: 'eventid', "
        "'sourcedoublecouple', 'sourceforce', 'sourcemomenttensor'"
    )

    # moment tensor is missing a component.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000",
        "components": "Z",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcemomenttensor' must be formatted as: "
        "'Mrr,Mtt,Mpp,Mrt,Mrp,Mtp'"
    )

    # moment tensor has an invalid component.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcemomenttensor": "100000,100000,100000,100000,100000,bogus",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcemomenttensor' must be formatted as: "
        "'Mrr,Mtt,Mpp,Mrt,Mrp,Mtp'"
    )

    # sourcedoublecouple is missing a component.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcedoublecouple' must be formatted as: "
        "'strike,dip,rake[,M0]'"
    )

    # sourcedoublecouple has an extra component.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,11,12",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcedoublecouple' must be formatted as: "
        "'strike,dip,rake[,M0]'"
    )

    # sourcedoublecouple has an invalid component.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,bogus",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourcedoublecouple' must be formatted as: "
        "'strike,dip,rake[,M0]'"
    )

    # sourceforce is missing a component.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourceforce": "100000,100000",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourceforce' must be formatted as: " "'Fr,Ft,Fp'"
    )

    # sourceforce has an invalid component.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourceforce": "100000,100000,bogus",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'sourceforce' must be formatted as: " "'Fr,Ft,Fp'"
    )

    # Seismic moment cannot be negative.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,-10",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == "Seismic moment must not be negative."

    # Funky phase offset setting.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,10",
        "dt": 0.1,
        "starttime": "P+!A",
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Parameter 'starttime' must be formatted as: 'Datetime "
        "String/Float/Phase+-Offset'"
    )

    # Mixing different source settings.
    params = {
        "sourcelatitude": -39,
        "sourcelongitude": 20,
        "sourcedepthinmeters": 300000,
        "sourcedoublecouple": "100000,100000,10,10",
        "sourceforce": "10,10,10",
        "dt": 0.1,
        "format": "miniseed",
        "network": "IU,B*",
        "station": "ANT*,ANM?",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Only one of these parameters can be given "
        "simultaneously: 'sourcedoublecouple', "
        "'sourceforce'"
    )

    if db.info.is_reciprocal:
        # Receiver depth must be at the surface for a reciprocal database.
        params = {
            "sourcelatitude": -39,
            "sourcelongitude": 20,
            "sourcedepthinmeters": 0,
            "sourcedoublecouple": "10,10,10,10",
            "receiverlatitude": 10,
            "receiverlongitude": 10,
            "receiverdepthinmeters": 10,
            "dt": 0.1,
            "format": "miniseed",
        }
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 400
        assert request.reason == (
            "Receiver must be at the surface for " "reciprocal databases."
        )

        # A too deep source depth raises.
        params = {
            "sourcelatitude": -39,
            "sourcelongitude": 20,
            "sourcedepthinmeters": 3e9,
            "sourcedoublecouple": "10,10,10,10",
            "receiverlatitude": 10,
            "receiverlongitude": 10,
            "dt": 0.1,
            "format": "miniseed",
        }
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 400
        assert request.reason == (
            "Source depth must be within the database "
            "range: 0.0 - 371000.0 meters."
        )

    # For a forward database the source depth must be equal to the database
    # depth!
    if db.info.is_reciprocal is False:
        params = {
            "sourcelatitude": -39,
            "sourcelongitude": 20,
            "sourcedepthinmeters": 14,
            "sourcedoublecouple": "10,10,10,10",
            "receiverlatitude": 10,
            "receiverlongitude": 10,
            "receiverdepthinmeters": 10,
            "dt": 0.1,
            "format": "miniseed",
        }
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 400
        assert request.reason == (
            "Source depth must be: %.1f km" % db.info.source_depth
        )


def test_sac_dist_header_edge_case(all_clients):
    """
    Regression test for https://github.com/krischer/instaseis/issues/55.
    """
    client = all_clients

    params = {
        "sourcelatitude": 0.0,
        "sourcelongitude": 0.0,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "receiverlatitude": 0.0,
        "receiverlongitude": 10.0,
        "format": "saczip",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st:
        assert tr.stats._format == "SAC"
        # Checked with http://www.fai.org/distance_calculation
        assert abs(tr.stats.sac.dist - 1113.194907792064) < 1e-4

    params = {
        "sourcelatitude": 0.0,
        "sourcelongitude": 100.0,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "receiverlatitude": 0.0,
        "receiverlongitude": 105.0,
        "format": "saczip",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st:
        assert tr.stats._format == "SAC"
        # Checked with http://www.fai.org/distance_calculation
        assert abs(tr.stats.sac.dist - 556.5974538960322) < 1e-4


def test_sac_headers(all_clients):
    """
    Tests the sac headers.
    """
    client = all_clients

    params = {
        "sourcelatitude": 1,
        "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "origintime": obspy.UTCDateTime(0),
        "scale": 0.5,
        "dt": 0.1,
        "starttime": "-1.5",
        "receiverlatitude": 22,
        "receiverlongitude": 44,
        "format": "saczip",
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200
    st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st:
        assert tr.stats._format == "SAC"
        # Instaseis will write SAC coordinates in WGS84!
        # Assert the station headers.
        assert (
            abs(tr.stats.sac.stla - geocentric_to_elliptic_latitude(22)) < 1e-6
        )
        assert abs(tr.stats.sac.stlo - 44) < 1e-6
        assert abs(tr.stats.sac.stdp - 0.0) < 1e-6
        assert abs(tr.stats.sac.stel - 0.0) < 1e-6
        # Assert the event parameters.
        assert (
            abs(tr.stats.sac.evla - geocentric_to_elliptic_latitude(1)) < 1e-6
        )
        assert abs(tr.stats.sac.evlo - 12) < 1e-6
        assert abs(tr.stats.sac.evdp - client.source_depth) < 1e-6
        assert abs(tr.stats.sac.mag - 4.151) < 1e-2
        # Thats what SPECFEM uses for a moment magnitude....
        assert tr.stats.sac.imagtyp == 55
        # Assume the reference time is the starttime.
        assert abs(tr.stats.sac.o - 1.5) < 1e-6

        # Distances, az and baz
        assert abs(tr.stats.sac.dist - 4180.436) < 1e-4
        assert abs(tr.stats.sac.az - 53.691647) < 1e-4
        assert abs(tr.stats.sac.baz - 240.38969) < 1e-4
        assert abs(tr.stats.sac.gcarc - 37.560081) < 1e-4

        # Test the "provenance".
        assert tr.stats.sac.kuser0 == "InstSeis"
        assert tr.stats.sac.kuser1 == "prem_iso"
        # Two test databases.
        assert tr.stats.sac.kt7 in ("A60945ec", "A0400524")
        assert tr.stats.sac.kt8.strip() == "I" + instaseis.__version__[:7]
        assert tr.stats.sac.user0 == 0.5

        # Test two more headers. Regression test for #45.
        assert tr.stats.sac.lpspol == 1
        assert tr.stats.sac.lcalda == 0


def test_sac_headers_azimuth_and_incidence(all_clients):
    """
    Tests azimuth and component inclination sac headers.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath)

    if db.info.components != "vertical and horizontal":
        return

    def _run_test(rec_latitude, rec_longitude, az_inc_map):
        params = {
            "sourcelatitude": 0,
            "sourcelongitude": 0,
            "sourcedepthinmeters": client.source_depth,
            "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
            "origintime": obspy.UTCDateTime(0),
            "scale": 0.5,
            "dt": 0.1,
            "starttime": "-1.5",
            "receiverlatitude": rec_latitude,
            "receiverlongitude": rec_longitude,
            "format": "saczip",
            "components": "ZNERT",
        }
        request = fetch_sync(client, _assemble_url("seismograms", **params))
        assert request.code == 200
        st = obspy.Stream()
        zip_obj = zipfile.ZipFile(request.buffer)
        for name in zip_obj.namelist():
            st += obspy.read(io.BytesIO(zip_obj.read(name)))

        for c, values in az_inc_map.items():
            h = st.select(component=c)[0].stats.sac
            # Assert azimuth an incidence angle.
            assert h.cmpaz == values[0]
            assert h.cmpinc == values[1]

    # The source is always at 0/0.
    # First value is azimuth, second incidence angle.
    _run_test(
        rec_latitude=0.0,
        rec_longitude=90.0,
        az_inc_map={
            "Z": (0.0, 0.0),
            "N": (0.0, 90.0),
            "E": (90.0, 90.0),
            "R": (90.0, 90.0),
            "T": (180.0, 90.0),
        },
    )

    _run_test(
        rec_latitude=0.0,
        rec_longitude=-90.0,
        az_inc_map={
            "Z": (0.0, 0.0),
            "N": (0.0, 90.0),
            "E": (90.0, 90.0),
            "R": (270.0, 90.0),
            "T": (0.0, 90.0),
        },
    )

    _run_test(
        rec_latitude=45.0,
        rec_longitude=0.0,
        az_inc_map={
            "Z": (0.0, 0.0),
            "N": (0.0, 90.0),
            "E": (90.0, 90.0),
            "R": (0.0, 90.0),
            "T": (90.0, 90.0),
        },
    )

    _run_test(
        rec_latitude=-45.0,
        rec_longitude=0.0,
        az_inc_map={
            "Z": (0.0, 0.0),
            "N": (0.0, 90.0),
            "E": (90.0, 90.0),
            "R": (180.0, 90.0),
            "T": (270.0, 90.0),
        },
    )


def test_sac_headers_azimuth_and_incidence_greens_route(all_greens_clients):
    """
    Same thing but for the greens route - in this case nothing should be set
    as its not really defined on non-geographic systems.
    """
    client = all_greens_clients

    params = {
        "sourcedepthinmeters": client.source_depth,
        "sourcedistanceindegrees": 20,
        "format": "saczip",
    }

    request = fetch_sync(client, _assemble_url("greens_function", **params))
    assert request.code == 200
    st = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st += obspy.read(io.BytesIO(zip_obj.read(name)))

    for tr in st:
        # Make sure they are not set.
        assert "cmpinc" not in tr.stats.sac
        assert "cmpaz" not in tr.stats.sac


def test_dt_settings(all_clients):
    """
    Cannot downsample nor sample to more than 100 Hz.
    """
    client = all_clients

    # Requesting exactly at the initial sampling rate works.
    params = {
        "sourcelatitude": 1,
        "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": client.info.dt,
        "receiverlatitude": 22,
        "receiverlongitude": 44,
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200

    # Request exactly at 100 Hz works.
    params = {
        "sourcelatitude": 1,
        "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": 0.01,
        "receiverlatitude": 22,
        "receiverlongitude": 44,
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200

    # Requesting at something in between works.
    params = {
        "sourcelatitude": 1,
        "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": 0.123,
        "receiverlatitude": 22,
        "receiverlongitude": 44,
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 200

    # Requesting a tiny bit above does not work.
    params = {
        "sourcelatitude": 1,
        "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": 0.009,
        "receiverlatitude": 22,
        "receiverlongitude": 44,
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "The smallest possible dt is 0.01. Please choose a smaller value and "
        "resample locally if needed."
    )

    # Requesting a tiny bit below also does not work.
    params = {
        "sourcelatitude": 1,
        "sourcelongitude": 12,
        "sourcedepthinmeters": client.source_depth,
        "sourcemomenttensor": "1E15,1E15,1E15,1E15,1E15,1E15",
        "dt": client.info.dt + 0.001,
        "receiverlatitude": 22,
        "receiverlongitude": 44,
    }
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "Cannot downsample. The sampling interval of the database is "
        "24.72485 seconds. Make sure to choose a smaller or equal one."
    )


def test_scale_parameter(all_clients):
    """
    Tests the `scale` parameter of the /seismograms route.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "format": "miniseed",
    }

    # Various sources.
    mt = {
        "mrr": "100000",
        "mtt": "200000",
        "mpp": "300000",
        "mrt": "400000",
        "mrp": "500000",
        "mtp": "600000",
    }
    mt_param = "100000,200000,300000,400000,500000,600000"

    # Retrieve reference.
    source = instaseis.Source(
        latitude=basic_parameters["sourcelatitude"],
        longitude=basic_parameters["sourcelongitude"],
        depth_in_m=0.0,
        origin_time=obspy.UTCDateTime(1900, 1, 1),
        **dict(
            (key[0] + "_" + key[1:], float(value))
            for (key, value) in mt.items()
        ),
    )
    receiver = instaseis.Receiver(
        latitude=basic_parameters["receiverlatitude"],
        longitude=basic_parameters["receiverlongitude"],
        depth_in_m=0.0,
        network="XX",
        station="SYN",
        location="SE",
    )
    st_db = db.get_seismograms(source=source, receiver=receiver)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

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
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # No change if a scale of 1 is set.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["scale"] = 1.0
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # No change if a scale of 3.5 is set.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["scale"] = 3.5
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

    for tr_server, tr_database in zip(st_server, st_db):
        tr_db = tr_database.copy()
        tr_db.data *= 3.5
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # Negative scale of -2.5.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["scale"] = -2.5
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    st_server = obspy.read(request.buffer)

    for tr_server, tr_database in zip(st_server, st_db):
        tr_db = tr_database.copy()
        tr_db.data *= -2.5
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(
            tr_server.data, tr_db.data, atol=1e-10 * tr_server.data.ptp()
        )

    # Scale of 0 should raise.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["scale"] = 0.0
    request = fetch_sync(client, _assemble_url("seismograms", **params))
    assert request.code == 400
    assert request.reason == (
        "A scale of zero means all seismograms have an amplitude "
        "of zero. No need to get it in the first place."
    )


def test_error_handling_custom_stf(all_clients):
    """
    Tests the error handling when passing a custom STF for the /seismograms
    service.
    """
    client = all_clients

    # The source time function file parsing happens first so we don't need
    # to worry about the other parameters for now.

    # Empty request.
    request = fetch_sync(
        client, _assemble_url("seismograms"), method="POST", body=b""
    )
    assert request.code == 400
    assert request.reason == (
        "The source time function must be given in the "
        "body of the POST request."
    )

    # Not a valid json file.
    request = fetch_sync(
        client, _assemble_url("seismograms"), method="POST", body=b"abcdefg"
    )
    assert request.code == 400
    assert request.reason == (
        "The body of the POST request is not a valid " "JSON file."
    )

    # Not a json file that's valid according to the schema.
    body = {"random": "things"}
    request = fetch_sync(
        client,
        _assemble_url("seismograms"),
        method="POST",
        body=json.dumps(body),
    )
    assert request.code == 400
    # This file has many problems thus the error might vary.
    assert request.reason.startswith("Validation Error in JSON file: ")

    valid_json = {
        "units": "moment_rate",
        "relative_origin_time_in_sec": 15.23,
        "sample_spacing_in_sec": 50.0,
        "data": [0.0, 4, 25, 5.6, 2.4, 0.0],
    }

    # Couple more wrong ones.
    body = copy.deepcopy(valid_json)
    body["units"] = "random"
    request = fetch_sync(
        client,
        _assemble_url("seismograms"),
        method="POST",
        body=json.dumps(body),
    )
    assert request.code == 400
    assert request.reason == (
        "Validation Error in JSON file: 'random' is not one of "
        "['moment_rate']"
    )

    body = copy.deepcopy(valid_json)
    body["sample_spacing_in_sec"] = -0.1
    request = fetch_sync(
        client,
        _assemble_url("seismograms"),
        method="POST",
        body=json.dumps(body),
    )
    assert request.code == 400
    assert request.reason == (
        "Validation Error in JSON file: -0.1 is less than the minimum of "
        "1e-05"
    )

    body = copy.deepcopy(valid_json)
    body["data"].append("hello")
    request = fetch_sync(
        client,
        _assemble_url("seismograms"),
        method="POST",
        body=json.dumps(body),
    )
    assert request.code == 400
    assert request.reason == (
        "Validation Error in JSON file: 'hello' is not of type 'number'"
    )

    # Does not start and end with zero.
    body = copy.deepcopy(valid_json)
    body["data"][0] = 0.3
    request = fetch_sync(
        client,
        _assemble_url("seismograms"),
        method="POST",
        body=json.dumps(body),
    )
    assert request.code == 400
    assert request.reason == (
        "STF data did not validate: Must begin and end with zero."
    )

    # The sample spacing must not be smaller than the database sampling.
    body = copy.deepcopy(valid_json)
    body["sample_spacing_in_sec"] = 10.0
    request = fetch_sync(
        client,
        _assemble_url("seismograms"),
        method="POST",
        body=json.dumps(body),
    )
    assert request.code == 400
    assert request.reason == (
        "'sample_spacing_in_sec' in the JSON file must not be smaller than "
        "the database dt [24.725 seconds]."
    )

    # Should raise for an all zeros array.
    body = copy.deepcopy(valid_json)
    body["data"] = [0, 0, 0, 0, 0, 0]
    request = fetch_sync(
        client,
        _assemble_url("seismograms"),
        method="POST",
        body=json.dumps(body),
    )
    assert request.code == 400
    assert (
        "All zero (or nearly all zero) source time functions don't "
        "make any sense."
    ) in request.reason


def test_custom_stf(all_clients):
    """
    Test the custom STF.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "format": "miniseed",
        "sourcemomenttensor": "100000,200000,300000,400000,500000,600000",
    }

    # First test: Just reconvolve with the sliprate of the database...that
    # should not change it at all!
    valid_json = {
        "units": "moment_rate",
        "relative_origin_time_in_sec": db.info.src_shift,
        "sample_spacing_in_sec": db.info.dt,
        "data": [float(_i) for _i in db.info.sliprate],
    }

    body = copy.deepcopy(valid_json)
    r = fetch_sync(
        client,
        _assemble_url("seismograms", **basic_parameters),
        method="POST",
        body=json.dumps(body),
    )
    assert r.code == 200
    st_custom_stf = obspy.read(r.buffer)

    r = fetch_sync(client, _assemble_url("seismograms", **basic_parameters))
    assert r.code == 200
    st_default = obspy.read(r.buffer)

    # Cut of the last couple of samples: the convolution requires a taper at
    # the end, thus samples at the end WILL be different.
    for tr in st_default + st_custom_stf:
        tr.data = tr.data[:-5]

    _compare_streams(st_custom_stf, st_default)

    # Now we try the same thing, but shift it one sample.
    body = copy.deepcopy(valid_json)
    body["relative_origin_time_in_sec"] = db.info.src_shift + db.info.dt
    r = fetch_sync(
        client,
        _assemble_url("seismograms", **basic_parameters),
        method="POST",
        body=json.dumps(body),
    )
    assert r.code == 200
    st_custom_stf = obspy.read(r.buffer)

    r = fetch_sync(client, _assemble_url("seismograms", **basic_parameters))
    assert r.code == 200
    st_default = obspy.read(r.buffer)

    with pytest.raises(AssertionError):
        _compare_streams(st_custom_stf, st_default)

    # We shifted the reference time of the custom stf seismograms one delta
    # to the right, thus the actuall seismogram will be shifted one delta to
    # the left!
    for tr in st_default:
        tr.data = tr.data[1:-5]
    for tr in st_custom_stf:
        tr.data = tr.data[:-6]

    # Now they should be identical again.
    _compare_streams(st_custom_stf, st_default)

    # Parameter "sourcewidth" not compatible with POST requests.
    body = copy.deepcopy(valid_json)
    body["relative_origin_time_in_sec"] = db.info.src_shift + db.info.dt
    r = fetch_sync(
        client,
        _assemble_url("seismograms", sourcewidth=1.0, **basic_parameters),
        method="POST",
        body=json.dumps(body),
    )
    assert r.code == 400
    assert r.reason == (
        "Parameter 'sourcewidth' is not allowed for POST " "requests."
    )


def test_gaussian_source_time_function_calculation():
    """
    Tests the calculation of a Gaussian source time function.
    """
    # Test the integral. More accurate for smaller deltas.
    _, y = util.get_gaussian_source_time_function(4, 1.2)
    assert np.isclose(simps(y, dx=1.2), 1.0, rtol=1e-2)
    _, y = util.get_gaussian_source_time_function(4, 1.0)
    assert np.isclose(simps(y, dx=1.0), 1.0, rtol=1e-3)
    _, y = util.get_gaussian_source_time_function(4, 0.1)
    assert np.isclose(simps(y, dx=0.1), 1.0, rtol=1e-6)
    _, y = util.get_gaussian_source_time_function(4, 0.01)
    assert np.isclose(simps(y, dx=0.01), 1.0, rtol=1e-7)

    # Test the offset. Always has to be larger then the chosen source width
    # and at a sample.
    assert util.get_gaussian_source_time_function(4, 1.0)[0] == 4.0
    assert util.get_gaussian_source_time_function(4, 1.1)[0] == 4.4
    assert util.get_gaussian_source_time_function(4, 1.2)[0] == 4.8
    assert util.get_gaussian_source_time_function(4, 2.0)[0] == 4.0

    # Test a known good solution.
    np.testing.assert_allclose(
        util.get_gaussian_source_time_function(4, 2.5)[1],
        [0.0, 1.089142e-3, 5.641895e-1, 1.089142e-3, 7.835433e-12, 0],
        rtol=1e-5,
    )


def test_sourcewidth_parameter(all_clients):
    """
    Tests the sourcewidth parameter.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "components": "".join(db.available_components),
        "format": "miniseed",
        "sourcemomenttensor": "100000,200000,300000,400000,500000,600000",
    }

    r = fetch_sync(
        client,
        _assemble_url("seismograms", sourcewidth=1.0, **basic_parameters),
    )
    assert r.code == 400
    assert r.reason == (
        "The sourcewidth must not be smaller than the mesh "
        "period of the database (100.000 seconds)."
    )

    r = fetch_sync(
        client,
        _assemble_url("seismograms", sourcewidth=601.0, **basic_parameters),
    )
    assert r.code == 400
    assert r.reason == "The sourcewidth must not be larger than 600 seconds."

    # This is unfortunately really hard to test - so we'll just take the FFT
    # of a normal and a reconvolved one and make sure the reconvolved one
    # has less energy.
    r = fetch_sync(client, _assemble_url("seismograms", **basic_parameters))
    assert r.code == 200
    st = obspy.read(r.buffer)
    assert len(st) >= 1

    r = fetch_sync(
        client,
        _assemble_url("seismograms", sourcewidth=200.0, **basic_parameters),
    )
    st_re = obspy.read(r.buffer)
    assert len(st_re) >= 1

    for comp in db.available_components:
        d = st.select(component=comp)[0].data
        d_re = st_re.select(component=comp)[0].data
        assert np.abs(np.fft.rfft(d)).sum() > np.abs(np.fft.rfft(d_re)).sum()


def test_cache_is_not_modified(all_clients):
    """
    Test for https://github.com/krischer/instaseis/issues/76.

    Make sure the cached values are not internally modified by requesting
    things multiple times multiple times and asserting they are identical
    every time.
    """
    client = all_clients
    db = instaseis.open_db(client.filepath)

    # Helper function to request something from the server multiple times and
    # making sure it stays the same.
    def request_multiple_times(url, params):
        request = fetch_sync(client, _assemble_url(url, **params))
        st_ref = obspy.read(request.buffer)

        for _i in range(3):
            request = fetch_sync(client, _assemble_url(url, **params))
            st_new = obspy.read(request.buffer)
            assert st_new == st_ref

    basic_parameters = {
        "sourcelatitude": 10,
        "sourcelongitude": 10,
        "sourcedepthinmeters": client.source_depth,
        "receiverlatitude": -10,
        "receiverlongitude": -10,
        "format": "miniseed",
    }

    # Various sources.
    mt_param = "100000,200000,300000,400000,500000,600000"
    sdr_param = "10,20,30,1000000"
    fs_param = "100000,200000,300000"

    time = obspy.UTCDateTime(2010, 1, 2, 3, 4, 5)

    # Moment tensor source.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
        params["receiverdepthinmeters"] = "0.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)
        params["receiverdepthinmeters"] = "55.0"
    params["origintime"] = str(time)
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"
    params["locationcode"] = "XX"
    request_multiple_times("seismograms", params)

    # Moment tensor source from strike, dip, and rake.
    params = copy.deepcopy(basic_parameters)
    params["sourcedoublecouple"] = sdr_param
    if client.is_reciprocal:
        params["sourcedepthinmeters"] = "5.0"
        params["receiverdepthinmeters"] = "0.0"
    else:
        params["sourcedepthinmeters"] = str(client.source_depth)
        params["receiverdepthinmeters"] = "55.0"
    params["origintime"] = str(time)
    params["networkcode"] = "BW"
    params["stationcode"] = "ALTM"
    params["locationcode"] = "XX"
    request_multiple_times("seismograms", params)

    # Force source only works for displ_only databases.
    if "displ_only" in client.filepath:
        params = copy.deepcopy(basic_parameters)
        params["sourceforce"] = fs_param
        params["sourcedepthinmeters"] = "5.0"
        params["receiverdepthinmeters"] = "0.0"
        params["origintime"] = str(time)
        params["networkcode"] = "BW"
        params["stationcode"] = "ALTM"
        params["locationcode"] = "XX"
        request_multiple_times("seismograms", params)

    # Now test other the other parameters.
    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["components"] = "".join(db.default_components[:1])
    request_multiple_times("seismograms", params)

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["units"] = "acceleration"
    request_multiple_times("seismograms", params)

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["units"] = "velocity"
    request_multiple_times("seismograms", params)

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["dt"] = "0.1"
    params["kernelwidth"] = "1"
    request_multiple_times("seismograms", params)

    params = copy.deepcopy(basic_parameters)
    params["sourcemomenttensor"] = mt_param
    params["dt"] = "0.1"
    params["kernelwidth"] = "2"
    params["units"] = "ACCELERATION"
    request_multiple_times("seismograms", params)

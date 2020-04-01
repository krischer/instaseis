#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the finite source route of the Instaseis server.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import copy
import io
import os
import zipfile

import obspy
import numpy as np
import pytest
from .tornado_testing_fixtures import *  # NOQA
from .tornado_testing_fixtures import _assemble_url

import instaseis

# Conditionally import mock either from the stdlib or as a separate library.
import sys

if sys.version_info[0] == 2:  # pragma: no cover
    import mock
else:  # pragma: no cover
    import unittest.mock as mock

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USGS_PARAM_FILE_1 = os.path.join(DATA, "nepal.param")
USGS_PARAM_FILE_2 = os.path.join(DATA, "chile.param")
USGS_PARAM_FILE_EMPTY = os.path.join(DATA, "empty.param")
USGS_PARAM_FILE_DEEP = os.path.join(DATA, "deep.param")
USGS_PARAM_FILE_AIR = os.path.join(DATA, "airquakes.param")
USGS_PARAM_FILE_LONG = os.path.join(DATA, "long_source.param")


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


def _parse_finite_source(filename):
    """
    Helper function parsing a finite source exactly how it is parsed in the
    finite source server route.
    """
    fs = instaseis.FiniteSource.from_usgs_param_file(
        filename, npts=10000, dt=0.1, trise_min=1.0
    )

    # All test databases are the same.
    dominant_period = 100.0
    dt = 24.724845445855724
    npts = 73

    # Things are now on purpose a bit different than in the actual
    # implementation to actually test something....
    shift = fs.time_shift
    for src in fs.pointsources:
        src.sliprate = np.concatenate(
            [np.zeros(10), src.sliprate, np.zeros(10)]
        )
        src.time_shift += 10 * dt - shift

    fs.lp_sliprate(freq=1.0 / dominant_period, zerophase=True)
    fs.resample_sliprate(dt=dt, nsamp=npts + 20)

    return fs


def test_triggering_random_error_during_parsing(reciprocal_clients):
    """
    Tests triggering a random error during the USGS param file parsing.
    """
    client = reciprocal_clients

    with io.open(__file__) as fh:
        body = fh.read()

    # default parameters
    params = {"receiverlongitude": 11, "receiverlatitude": 22}

    with mock.patch(
        "instaseis.source.FiniteSource" ".from_usgs_param_file"
    ) as p:
        p.side_effect = ValueError("random crash")
        request = fetch_sync(
            client,
            _assemble_url("finite_source", **params),
            method="POST",
            body=body,
        )

    assert request.code == 400
    assert request.reason == (
        "Could not parse the body contents. " "Incorrect USGS param file?"
    )


def test_sending_non_usgs_file(reciprocal_clients):
    """
    Tests error if a non-USGS file is sent.
    """
    client = reciprocal_clients

    with io.open(__file__) as fh:
        body = fh.read()

    # default parameters
    params = {"receiverlongitude": 11, "receiverlatitude": 22}
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The body contents could not be parsed as an "
        "USGS param file due to: "
        "Not a valid USGS param file."
    )


@pytest.mark.parametrize("usgs_param", [USGS_PARAM_FILE_1, USGS_PARAM_FILE_2])
def test_finite_source_retrieval(reciprocal_clients, usgs_param):
    """
    Tests if the finite sources requested from the server are identical to
    the one requested with the local instaseis client with some required
    tweaks.
    """
    client = reciprocal_clients

    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "receiverdepthinmeters": 0,
        "format": "miniseed",
    }

    with io.open(usgs_param, "rb") as fh:
        body = fh.read()

    # default parameters
    params = copy.deepcopy(basic_parameters)
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_server = obspy.read(request.buffer)
    for tr in st_server:
        assert tr.stats._format == "MSEED"

    # Parse the finite source.
    fs = _parse_finite_source(usgs_param)
    rec = instaseis.Receiver(
        latitude=22, longitude=11, network="XX", station="SYN", location="SE"
    )

    st_db = db.get_seismograms_finite_source(sources=fs, receiver=rec)
    # The origin time is the time of the first sample in the route.
    for tr in st_db:
        # Cut away the first ten samples as they have been previously added.
        tr.data = tr.data[10:]
        tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1)

    for tr_db, tr_server in zip(st_db, st_server):
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        del tr_server.stats._format
        del tr_server.stats.mseed

        assert tr_db.stats == tr_server.stats
        np.testing.assert_allclose(tr_db.data, tr_server.data)

    # Once again but this time request a SAC file.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st_server:
        assert tr.stats._format == "SAC"

    for tr_db, tr_server in zip(st_db, st_server):
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        del tr_server.stats._format
        del tr_server.stats.sac

        assert tr_db.stats == tr_server.stats
        np.testing.assert_allclose(tr_db.data, tr_server.data)

    # One with a label.
    params = copy.deepcopy(basic_parameters)
    params["label"] = "random_things"
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200

    cd = request.headers["Content-Disposition"]
    assert cd.startswith("attachment; filename=random_things_")
    assert cd.endswith(".mseed")

    # One simulating a crash in the underlying function.
    params = copy.deepcopy(basic_parameters)

    with mock.patch(
        "instaseis.database_interfaces.base_instaseis_db"
        ".BaseInstaseisDB.get_seismograms_finite_source"
    ) as p:
        p.side_effect = ValueError("random crash")
        request = fetch_sync(
            client,
            _assemble_url("finite_source", **params),
            method="POST",
            body=body,
        )

    assert request.code == 400
    assert request.reason == (
        "Could not extract finite source seismograms. "
        "Make sure, the parameters are valid, and the "
        "depth settings are correct."
    )

    # Simulating a logic error that should not be able to happen.
    params = copy.deepcopy(basic_parameters)
    with mock.patch(
        "instaseis.database_interfaces.base_instaseis_db"
        ".BaseInstaseisDB.get_seismograms_finite_source"
    ) as p:
        # Longer than the database returned stream thus the endtime is out
        # of bounds.
        st = obspy.read()

        p.return_value = st
        request = fetch_sync(
            client,
            _assemble_url("finite_source", **params),
            method="POST",
            body=body,
        )

    assert request.code == 500
    assert request.reason.startswith(
        "Endtime larger than the extracted " "endtime"
    )

    # One more with resampling parameters and different units.
    params = copy.deepcopy(basic_parameters)
    # We must have a sampling rate that cleanly fits in the existing one,
    # otherwise we cannot fake the cutting.
    dt_new = 24.724845445855724 / 10
    params["dt"] = dt_new
    params["kernelwidth"] = 2
    params["units"] = "acceleration"

    st_db = db.get_seismograms_finite_source(
        sources=fs, receiver=rec, dt=dt_new, kernelwidth=2, kind="acceleration"
    )
    # The origin time is the time of the first sample in the route.
    for tr in st_db:
        # Cut away the first ten samples as they have been previously added.
        tr.data = tr.data[100:]
        tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1)

    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    # Cut some parts in the middle to avoid any potential boundary effects.
    st_db.trim(
        obspy.UTCDateTime(1900, 1, 1, 0, 4),
        obspy.UTCDateTime(1900, 1, 1, 0, 14),
    )
    st_server.trim(
        obspy.UTCDateTime(1900, 1, 1, 0, 4),
        obspy.UTCDateTime(1900, 1, 1, 0, 14),
    )

    for tr_db, tr_server in zip(st_db, st_server):
        # Sample spacing and times are very similar but not identical due to
        # floating point inaccuracies in the arithmetics.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        np.testing.assert_allclose(
            tr_server.stats.starttime.timestamp,
            tr_db.stats.starttime.timestamp,
        )
        tr_server.stats.delta = tr_db.stats.delta
        tr_server.stats.starttime = tr_db.stats.starttime
        del tr_server.stats._format
        del tr_server.stats.mseed
        del tr_server.stats.processing
        del tr_db.stats.processing

        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)

        assert tr_db.stats == tr_server.stats
        np.testing.assert_allclose(
            tr_db.data, tr_server.data, rtol=1e-7, atol=tr_db.data.ptp() * 1e-7
        )

    # Testing network and station code parameters.
    # Default values.
    params = copy.deepcopy(basic_parameters)
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_server = obspy.read(request.buffer)
    for tr in st_server:
        assert tr.stats.network == "XX"
        assert tr.stats.station == "SYN"
        assert tr.stats.location == "SE"

    # Setting all three.
    params = copy.deepcopy(basic_parameters)
    params["networkcode"] = "AA"
    params["stationcode"] = "BB"
    params["locationcode"] = "CC"
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_server = obspy.read(request.buffer)
    for tr in st_server:
        assert tr.stats.network == "AA"
        assert tr.stats.station == "BB"
        assert tr.stats.location == "CC"

    # Setting only the location code.
    params = copy.deepcopy(basic_parameters)
    params["locationcode"] = "AA"
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_server = obspy.read(request.buffer)
    for tr in st_server:
        assert tr.stats.network == "XX"
        assert tr.stats.station == "SYN"
        assert tr.stats.location == "AA"

    # Test the scale parameter.
    params = copy.deepcopy(basic_parameters)
    params["scale"] = 33.33
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_server = obspy.read(request.buffer)
    for tr in st_server:
        assert tr.stats._format == "MSEED"

    # Parse the finite source.
    fs = _parse_finite_source(usgs_param)
    rec = instaseis.Receiver(
        latitude=22, longitude=11, network="XX", station="SYN", location="SE"
    )

    st_db = db.get_seismograms_finite_source(sources=fs, receiver=rec)
    # The origin time is the time of the first sample in the route.
    for tr in st_db:
        # Multiply with scale parameter.
        tr.data *= 33.33
        # Cut away the first ten samples as they have been previously added.
        tr.data = tr.data[10:]
        tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1)

    for tr_db, tr_server in zip(st_db, st_server):
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        del tr_server.stats._format
        del tr_server.stats.mseed

        assert tr_db.stats == tr_server.stats
        np.testing.assert_allclose(
            tr_db.data, tr_server.data, atol=1e-6 * tr_db.data.ptp()
        )


@pytest.mark.parametrize("usgs_param", [USGS_PARAM_FILE_1, USGS_PARAM_FILE_2])
def test_more_complex_queries(reciprocal_clients_all_callbacks, usgs_param):
    """
    These are not exhaustive tests but test that the queries do something.
    Elsewhere they are tested in more details.

    Test phase relative offsets.

    + must be encoded with %2B
    - must be encoded with %2D
    """
    client = reciprocal_clients_all_callbacks
    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "components": "".join(db.available_components),
        "format": "miniseed",
    }

    with io.open(usgs_param, "rb") as fh:
        body = fh.read()

    # default parameters
    params = copy.deepcopy(basic_parameters)
    params["dt"] = 2
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st = obspy.read(request.buffer)

    # Now request one starting ten seconds later.
    params = copy.deepcopy(basic_parameters)
    params["starttime"] = 10
    params["dt"] = 2
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_2 = obspy.read(request.buffer)

    assert st[0].stats.starttime + 10 == st_2[0].stats.starttime

    # The rest of data should still be identical.
    np.testing.assert_allclose(st.slice(starttime=10)[0].data, st_2[0].data)

    # Try with the endtime.
    params = copy.deepcopy(basic_parameters)
    params["endtime"] = 20
    params["dt"] = 2
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_2 = obspy.read(request.buffer)

    assert st_2[0].stats.endtime == st[0].stats.starttime + 18

    # The rest of data should still be identical.
    np.testing.assert_allclose(
        st.slice(endtime=st[0].stats.starttime + 18)[0].data, st_2[0].data
    )

    # Phase relative start and endtimes.
    params = copy.deepcopy(basic_parameters)
    params["starttime"] = "P%2D5"
    params["endtime"] = "P%2B5"
    params["dt"] = 2
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_2 = obspy.read(request.buffer)

    # Just make sure they actually do something.
    assert st_2[0].stats.starttime > st[0].stats.starttime
    assert st_2[0].stats.endtime < st[0].stats.endtime

    # Mixing things
    params = copy.deepcopy(basic_parameters)
    params["starttime"] = "P%2D5"
    params["endtime"] = "50"
    params["dt"] = 2
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_2 = obspy.read(request.buffer)

    # Just make sure they actually do something.
    assert st_2[0].stats.starttime > st[0].stats.starttime
    assert st_2[0].stats.endtime < st[0].stats.endtime

    if "Z" not in db.available_components:
        return

    # Network and station searches.
    params = {
        "network": "IU,B*",
        "station": "ANT*,ANM?",
        "components": "Z",
        "format": "miniseed",
    }
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 200
    st_2 = obspy.read(request.buffer)

    assert sorted(tr.id for tr in st_2) == ["IU.ANMO..LXZ", "IU.ANTO..LXZ"]


@pytest.mark.parametrize("usgs_param", [USGS_PARAM_FILE_1, USGS_PARAM_FILE_2])
def test_various_failure_conditions(
    reciprocal_clients_all_callbacks, usgs_param
):
    """
    Tests some failure conditions.
    """
    client = reciprocal_clients_all_callbacks

    basic_parameters = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "components": "Z",
        "format": "miniseed",
    }

    with io.open(usgs_param, "rb") as fh:
        body = fh.read()

    # Starttime too large.
    params = copy.deepcopy(basic_parameters)
    params["starttime"] = 200000
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The `starttime` must be before the seismogram " "ends."
    )

    # Starttime too early.
    params = copy.deepcopy(basic_parameters)
    params["starttime"] = -10000
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The seismogram can start at the maximum one "
        "hour before the origin time."
    )

    # Endtime too small.
    params = copy.deepcopy(basic_parameters)
    params["endtime"] = -200000
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The end time of the seismograms lies outside " "the allowed range."
    )

    # Useless phase relative times. pdiff does not exist at the epicentral
    # range of the example files.
    params = copy.deepcopy(basic_parameters)
    params["starttime"] = "Pdiff%2D5"
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "No seismograms found for the given phase "
        "relative offsets. This could either be due to "
        "the chosen phase not existing for the specific "
        "source-receiver geometry or arriving too "
        "late/with too large offsets if the database is "
        "not long enough."
    )

    # Scale of zero.
    params = copy.deepcopy(basic_parameters)
    params["scale"] = 0.0
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason.startswith("A scale of zero means")

    # Invalid receiver coordinates.
    params = copy.deepcopy(basic_parameters)
    params["receiverlatitude"] = 1e9
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "Could not construct receiver with passed "
        "parameters. Check parameters for sanity."
    )

    # Invalid receiver coordinates based on a station coordinates query.
    params = copy.deepcopy(basic_parameters)
    del params["receiverlatitude"]
    del params["receiverlongitude"]
    params["network"] = "XX"
    params["station"] = "DUMMY"
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "Station coordinate query returned invalid " "coordinates."
    )

    # Coordinates not found
    params = copy.deepcopy(basic_parameters)
    del params["receiverlatitude"]
    del params["receiverlongitude"]
    params["network"] = "UN"
    params["station"] = "KNOWN"
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 404
    assert request.reason == "No coordinates found satisfying the query."


def test_uploading_empty_usgs_file(reciprocal_clients):
    """
    Tests uploading an empty usgs file.
    """
    client = reciprocal_clients

    params = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "components": "Z",
        "format": "miniseed",
    }

    with io.open(USGS_PARAM_FILE_EMPTY, "rb") as fh:
        body = fh.read()

    # Starttime too large.
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The body contents could not be parsed as an USGS param file due to: "
        "No point sources found in the file."
    )


def test_uploading_deep_usgs_file(reciprocal_clients):
    """
    Tests uploading a usgs file that has sources that are too deep for the
    database.
    """
    client = reciprocal_clients

    params = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "components": "Z",
        "format": "miniseed",
    }

    with io.open(USGS_PARAM_FILE_DEEP, "rb") as fh:
        body = fh.read()

    # Starttime too large.
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The deepest point source in the given finite "
        "source is 1000.9 km deep. The database only "
        "has a depth range from 0.0 km to 371.0 km."
    )


def test_uploading_usgs_file_with_airquakes(reciprocal_clients):
    """
    Tests uploading a usgs file that has sources that are above the planet
    radius.
    """
    client = reciprocal_clients

    params = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "components": "Z",
        "format": "miniseed",
    }

    with io.open(USGS_PARAM_FILE_AIR, "rb") as fh:
        body = fh.read()

    # Starttime too large.
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The shallowest point source in the given "
        "finite source is -0.9 km deep. The database "
        "only has a depth range from 0.0 km to "
        "371.0 km."
    )


def test_uploading_usgs_file_with_long_rise_or_fall_times(reciprocal_clients):
    """
    Tests uploading a usgs file that has too long rise or fall times.
    radius.
    """
    client = reciprocal_clients

    params = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "components": "Z",
        "format": "miniseed",
    }

    with io.open(USGS_PARAM_FILE_LONG, "rb") as fh:
        body = fh.read()

    # Starttime too large.
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The body contents could not be parsed as an "
        "USGS param file due to: Rise + fall time are "
        "longer than the total length of calculated "
        "slip. Please use more samples."
    )


def test_uploading_file_with_too_many_sources(reciprocal_clients):
    """
    Tests uploading a file with too many point sources.
    """
    client = reciprocal_clients

    # Artificially limit the number of allowed sources.
    client.application.max_size_of_finite_sources = 17

    params = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "components": "Z",
        "format": "miniseed",
    }

    with io.open(USGS_PARAM_FILE_1, "rb") as fh:
        body = fh.read()

    # Starttime too large.
    request = fetch_sync(
        client,
        _assemble_url("finite_source", **params),
        method="POST",
        body=body,
    )
    assert request.code == 400
    assert request.reason == (
        "The server only allows finite sources with at "
        "most 17 points sources. The source in question "
        "has 121 points."
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The tornado.testing Async stuff packaged as fixtures for use with py.test

Originally from https://gist.github.com/robcowie/7843633; modified for the
instaseis server.
"""
from collections import OrderedDict
import inspect
import os
import pytest
import re
import responses

from tornado.httpserver import HTTPServer
from tornado.httpclient import AsyncHTTPClient
from tornado.ioloop import IOLoop
from tornado.testing import bind_unused_port

from obspy.taup import TauPyModel
from obspy import geodetics

import instaseis
from instaseis.server.app import get_application
from instaseis.database_interfaces import find_and_open_files


# Most generic way to get the data folder path.
DATA = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    "data",
)

MODEL = TauPyModel("ak135")


def _assemble_url(route, **kwargs):
    """
    Helper function.
    """
    url = "/%s?" % route
    url += "&".join("%s=%s" % (key, value) for key, value in kwargs.items())
    return url


# All test databases. Fix order so tests can be executed in parallel.
DBS = OrderedDict()
DBS["db_bwd_displ_only"] = os.path.join(DATA, "100s_db_bwd_displ_only")
DBS["db_bwd_strain_only"] = os.path.join(DATA, "100s_db_bwd_strain_only")
DBS["db_fwd"] = os.path.join(DATA, "100s_db_fwd")
DBS["db_fwd_deep"] = os.path.join(DATA, "100s_db_fwd_deep")

_CONFIG_DBS = instaseis._test_dbs

# Add all automatically created repacked databases to the server test suite.
for name, path in _CONFIG_DBS["databases"].items():
    DBS[name] = path


def event_info_mock_callback(event_id):
    """
    Mock the event info callback for the purpose of testing.
    """
    if event_id == "B071791B":
        return {
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
    # Half the information is missing.
    elif event_id == "invalid_event":
        return {
            "m_rr": -58000000000000000,
            "m_pp": -20100000000000000,
            "m_rp": 108100000000000000,
            "latitude": -3.8,
            "origin_time": "1991-07-17T16:41:33.100000Z",
        }
    else:
        raise ValueError


def station_coordinates_mock_callback(networks, stations):
    """
    Mock station coordinates callback for the purpose of testing.

    Return a single station for networks=["IU"] and stations=["ANMO"],
    two stations for networks=["IU", "B*"] and stations=["ANT*", "ANM?"] and no
    stations for all other combinations.
    """
    if networks == ["IU"] and stations == ["ANMO"]:
        return [
            {
                "latitude": 34.94591,
                "longitude": -106.4572,
                "network": "IU",
                "station": "ANMO",
            }
        ]
    elif networks == ["IU", "B*"] and stations == ["ANT*", "ANM?"]:
        return [
            {
                "latitude": 39.868,
                "longitude": 32.7934,
                "network": "IU",
                "station": "ANTO",
            },
            {
                "latitude": 34.94591,
                "longitude": -106.4572,
                "network": "IU",
                "station": "ANMO",
            },
        ]
    # Invalid coordinates!
    elif networks == ["XX"] and stations == ["DUMMY"]:
        return [
            {
                "latitude": 3e9,
                "longitude": -106.4572,
                "network": "XX",
                "station": "DUMMY",
            }
        ]
    else:
        return []


def get_travel_time(
    sourcelatitude,
    sourcelongitude,
    sourcedepthinmeters,
    receiverlatitude,
    receiverlongitude,
    receiverdepthinmeters,
    phase_name,
    db_info,
):
    """
    Fully working travel time callback implementation.
    """
    if receiverdepthinmeters:
        raise ValueError(
            "This travel time implementation cannot calculate "
            "buried receivers."
        )

    great_circle_distance = geodetics.locations2degrees(
        sourcelatitude, sourcelongitude, receiverlatitude, receiverlongitude
    )

    try:
        tts = MODEL.get_travel_times(
            source_depth_in_km=sourcedepthinmeters / 1000.0,
            distance_in_degree=great_circle_distance,
            phase_list=[phase_name],
        )
    except Exception as e:
        raise ValueError(str(e))

    if not tts:
        return None

    # For any phase, return the first time.
    return tts[0].time


@pytest.fixture
def io_loop(request):
    """Create an instance of the `tornado.ioloop.IOLoop` for each test case.
    """
    io_loop = IOLoop()
    io_loop.make_current()

    def _close():
        io_loop.clear_current()
        io_loop.close(all_fds=True)

    request.addfinalizer(_close)
    return io_loop


def create_async_client(
    io_loop,
    request,
    path,
    station_coordinates_callback=None,
    event_info_callback=None,
    travel_time_callback=None,
):
    application = get_application()
    application.db = find_and_open_files(path=path)
    application.station_coordinates_callback = station_coordinates_callback
    application.event_info_callback = event_info_callback
    application.travel_time_callback = travel_time_callback
    application.max_size_of_finite_sources = 1000

    # Build server.
    sock, port = bind_unused_port()
    server = HTTPServer(application)
    server.add_sockets([sock])

    def _stop():
        server.stop()

        if hasattr(server, "close_all_connections"):
            io_loop.run_sync(server.close_all_connections, timeout=30)

    request.addfinalizer(_stop)

    # Build client.
    client = AsyncHTTPClient()
    client.loop = io_loop

    def _close():
        client.close()

    request.addfinalizer(_close)

    client.application = application
    client.filepath = path
    client.port = port

    # Flag to help deal with forward/backwards databases.
    b = os.path.basename(path)
    if "bwd" in b or "horizontal_only" in b or "vertical_only" in b:
        client.is_reciprocal = True
        client.source_depth = 0.0
    else:
        client.is_reciprocal = False
        client.source_depth = application.db.info.source_depth * 1000
    client.info = application.db.info
    return client


@pytest.fixture(params=list(DBS.values()))
def all_clients(io_loop, request):
    """
    Fixture returning all clients!
    """
    return create_async_client(
        io_loop, request, request.param, station_coordinates_callback=None
    )


@pytest.fixture(
    params=[
        _i
        for _i in list(DBS.values())
        if (
            "db_bwd" in _i
            and "horizontal_only" not in _i
            and "vertical_only" not in _i
        )
    ]
)
def all_greens_clients(io_loop, request):
    """
    Fixture returning all clients compatible with Green's functions!
    """
    return create_async_client(
        io_loop, request, request.param, station_coordinates_callback=None
    )


@pytest.fixture(params=[_i for _i in list(DBS.values()) if "db_bwd" in _i])
def reciprocal_clients(io_loop, request):
    """
    Fixture returning all reciprocal clients!
    """
    return create_async_client(
        io_loop, request, request.param, station_coordinates_callback=None
    )


@pytest.fixture(params=list(DBS.values()))
def all_clients_station_coordinates_callback(io_loop, request):
    """
    Fixture returning all with a station coordinates callback.
    """
    return create_async_client(
        io_loop,
        request,
        request.param,
        station_coordinates_callback=station_coordinates_mock_callback,
    )


@pytest.fixture(params=list(DBS.values()))
def all_clients_event_callback(io_loop, request):
    """
    Fixture returning all with a event info callback.
    """
    return create_async_client(
        io_loop,
        request,
        request.param,
        event_info_callback=event_info_mock_callback,
    )


@pytest.fixture(params=list(DBS.values()))
def all_clients_ttimes_callback(io_loop, request):
    """
    Fixture returning all clients with a travel time callback.
    """
    return create_async_client(
        io_loop, request, request.param, travel_time_callback=get_travel_time
    )


@pytest.fixture(
    params=[
        _i
        for _i in list(DBS.values())
        if (
            "db_bwd" in _i
            and "horizontal_only" not in _i
            and "vertical_only" not in _i
        )
    ]
)
def all_greens_clients_ttimes_callback(io_loop, request):
    """
    Fixture returning all clients compatible with Green's functions!
    """
    return create_async_client(
        io_loop, request, request.param, travel_time_callback=get_travel_time
    )


@pytest.fixture(
    params=[
        _i for _i in list(DBS.values()) if ("db_bwd" in _i or "_only_" in _i)
    ]
)
def reciprocal_clients_all_callbacks(io_loop, request):
    """
    Fixture returning reciprocal clients with all callbacks.
    """
    return create_async_client(
        io_loop,
        request,
        request.param,
        station_coordinates_callback=station_coordinates_mock_callback,
        event_info_callback=event_info_mock_callback,
        travel_time_callback=get_travel_time,
    )


@pytest.fixture(params=list(DBS.values()))
def all_clients_all_callbacks(io_loop, request):
    """
    Fixture returning all clients with all callbacks.
    """
    return create_async_client(
        io_loop,
        request,
        request.param,
        station_coordinates_callback=station_coordinates_mock_callback,
        event_info_callback=event_info_mock_callback,
        travel_time_callback=get_travel_time,
    )


def _add_callback(client):
    # Convert the async callback to a sync one so the responses library can
    # work with it.
    def request_callback(request):
        async def f():
            response = await client.fetch(request.url)
            return response

        r = client.io_loop.run_sync(f)

        return (r.code, r.headers, r.body)

    pattern = re.compile(r"http://localhost.*")
    responses.add_callback(
        responses.GET,
        pattern,
        callback=request_callback,
        content_type="application/octet_stream",
    )


@pytest.fixture(params=list(DBS.values()))
@responses.activate
def all_remote_dbs(io_loop, request):
    client = create_async_client(io_loop, request, request.param, None)

    _add_callback(client)

    db = instaseis.open_db("http://localhost:%i" % client.port)
    db._client = client
    return db

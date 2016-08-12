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
import socket
import sys

from tornado import netutil
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.testing import AsyncHTTPClient
from tornado.util import raise_exc_info

from obspy.taup import TauPyModel
from obspy import geodetics

import instaseis
from instaseis.server.app import get_application
from instaseis.database_interfaces import find_and_open_files


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")

MODEL = TauPyModel("ak135")


def _assemble_url(route, **kwargs):
    """
    Helper function.
    """
    url = "/%s?" % route
    url += "&".join("%s=%s" % (key, value) for key, value in kwargs.items())
    return url


class AsyncClient(object):
    """
    A port of parts of AsyncTestCase. See tornado.testing.py:275
    """
    def __init__(self, httpserver, httpclient):
        self.__stopped = False
        self.__running = False
        self.__stop_args = None
        self.__failure = None
        self.httpserver = httpserver
        self.httpclient = httpclient

    def _get_port(self):
        # This seems a bit fragile. How else to get the dynamic port number?
        return list(self.httpserver._sockets.values())[0].getsockname()[1]

    def fetch(self, path, use_gzip=False, **kwargs):
        port = self._get_port()
        url = u'%s://localhost:%s%s' % ('http', port, path)
        self.httpclient.fetch(url, self.stop, decompress_response=use_gzip,
                              **kwargs)
        return self.wait()

    @property
    def io_loop(self):
        # We're using a singleton ioloop throughout
        return IOLoop.instance()

    def stop(self, _arg=None, **kwargs):
        """
        Stops the `.IOLoop`, causing one pending (or future) call to `wait()`
        to return.
        """
        assert _arg is None or not kwargs
        self.__stop_args = kwargs or _arg
        if self.__running:
            self.io_loop.stop()
            self.__running = False
        self.__stopped = True

    def __rethrow(self):  # pragma: no cover
        if self.__failure is not None:
            failure = self.__failure
            self.__failure = None
            raise_exc_info(failure)

    def wait(self, condition=None, timeout=None):
        if timeout is None:
            timeout = 30

        if not self.__stopped:
            if timeout:  # pragma: no cover
                def timeout_func():
                    try:
                        raise self.failureException(
                            'Async operation timed out after %s seconds' %
                            timeout)
                    except Exception:
                        self.__failure = sys.exc_info()
                    self.stop()
                self.__timeout = self.io_loop.add_timeout(
                    self.io_loop.time() + timeout, timeout_func)
            while True:
                self.__running = True
                self.io_loop.start()
                if (self.__failure is not None or
                        condition is None or condition()):
                    break
            if self.__timeout is not None:
                self.io_loop.remove_timeout(self.__timeout)
                self.__timeout = None
        assert self.__stopped
        self.__stopped = False
        self.__rethrow()
        result = self.__stop_args
        self.__stop_args = None
        return result


def bind_unused_port():
    """
    Binds a server socket to an available port on localhost.
    Returns a tuple (socket, port).
    """
    sock = netutil.bind_sockets(None, 'localhost', family=socket.AF_INET)[0]
    port = sock.getsockname()[1]
    return sock, port


# All test databases. Fix order so tests can be executed in parallel.
DBS = OrderedDict()
DBS["db_bwd_displ_only"] = os.path.join(DATA, "100s_db_bwd_displ_only")
DBS["db_bwd_strain_only"] = os.path.join(DATA, "100s_db_bwd_strain_only")
DBS["db_fwd"] = os.path.join(DATA, "100s_db_fwd")
DBS["db_fwd_deep"] = os.path.join(DATA, "100s_db_fwd_deep")

# Add all automatically created repacked databases to the server test suite.
for name, path in pytest.config.dbs["databases"].items():
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
            "origin_time": "1991-07-17T16:41:33.100000Z"}
    # Half the information is missing.
    elif event_id == "invalid_event":
        return {
            "m_rr": -58000000000000000,
            "m_pp": -20100000000000000,
            "m_rp": 108100000000000000,
            "latitude": -3.8,
            "origin_time": "1991-07-17T16:41:33.100000Z"}
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
        return [{
            "latitude": 34.94591,
            "longitude": -106.4572,
            "network": "IU",
            "station": "ANMO"}]
    elif networks == ["IU", "B*"] and stations == ["ANT*", "ANM?"]:
        return [{
            "latitude": 39.868,
            "longitude": 32.7934,
            "network": "IU",
            "station": "ANTO"
         }, {
            "latitude": 34.94591,
            "longitude": -106.4572,
            "network": "IU",
            "station": "ANMO"}]
    # Invalid coordinates!
    elif networks == ["XX"] and stations == ["DUMMY"]:
        return [{
            "latitude": 3E9,
            "longitude": -106.4572,
            "network": "XX",
            "station": "DUMMY"}]
    else:
        return []


def get_travel_time(sourcelatitude, sourcelongitude, sourcedepthinmeters,
                    receiverlatitude, receiverlongitude,
                    receiverdepthinmeters, phase_name, db_info):
    """
    Fully working travel time callback implementation.
    """
    if receiverdepthinmeters:
        raise ValueError("This travel time implementation cannot calculate "
                         "buried receivers.")

    great_circle_distance = geodetics.locations2degrees(
        sourcelatitude, sourcelongitude, receiverlatitude, receiverlongitude)

    try:
        tts = MODEL.get_travel_times(
            source_depth_in_km=sourcedepthinmeters / 1000.0,
            distance_in_degree=great_circle_distance,
            phase_list=[phase_name])
    except Exception as e:
        raise ValueError(str(e))

    if not tts:
        return None

    # For any phase, return the first time.
    return tts[0].time


def create_async_client(path, station_coordinates_callback=None,
                        event_info_callback=None,
                        travel_time_callback=None):
    application = get_application()
    application.db = find_and_open_files(path=path)
    application.station_coordinates_callback = station_coordinates_callback
    application.event_info_callback = event_info_callback
    application.travel_time_callback = travel_time_callback
    application.max_size_of_finite_sources = 1000
    # Build server as in testing:311
    sock, port = bind_unused_port()
    server = HTTPServer(application, io_loop=IOLoop.instance())
    server.add_sockets([sock])
    client = AsyncClient(server, AsyncHTTPClient())
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
def all_clients(request):
    """
    Fixture returning all clients!
    """
    return create_async_client(request.param,
                               station_coordinates_callback=None)


@pytest.fixture(params=[_i for _i in list(DBS.values()) if (
        "db_bwd" in _i and
        "horizontal_only" not in _i and
        "vertical_only" not in _i)])
def all_greens_clients(request):
    """
    Fixture returning all clients compatible with Green's functions!
    """
    return create_async_client(request.param,
                               station_coordinates_callback=None)


@pytest.fixture(params=[_i for _i in list(DBS.values()) if "db_bwd" in _i])
def reciprocal_clients(request):
    """
    Fixture returning all reciprocal clients!
    """
    return create_async_client(request.param,
                               station_coordinates_callback=None)


@pytest.fixture(params=list(DBS.values()))
def all_clients_station_coordinates_callback(request):
    """
    Fixture returning all with a station coordinates callback.
    """
    return create_async_client(
        request.param,
        station_coordinates_callback=station_coordinates_mock_callback)


@pytest.fixture(params=list(DBS.values()))
def all_clients_event_callback(request):
    """
    Fixture returning all with a event info callback.
    """
    return create_async_client(
        request.param,
        event_info_callback=event_info_mock_callback)


@pytest.fixture(params=list(DBS.values()))
def all_clients_ttimes_callback(request):
    """
    Fixture returning all clients with a travel time callback.
    """
    return create_async_client(
        request.param,
        travel_time_callback=get_travel_time)


@pytest.fixture(params=[_i for _i in list(DBS.values()) if (
        "db_bwd" in _i and
        "horizontal_only" not in _i and
        "vertical_only" not in _i)])
def all_greens_clients_ttimes_callback(request):
    """
    Fixture returning all clients compatible with Green's functions!
    """
    return create_async_client(
        request.param,
        travel_time_callback=get_travel_time)


@pytest.fixture(params=[_i for _i in list(DBS.values()) if
                        ("db_bwd" in _i or "_only_" in _i)])
def reciprocal_clients_all_callbacks(request):
    """
    Fixture returning reciprocal clients with all callbacks.
    """
    return create_async_client(
        request.param,
        station_coordinates_callback=station_coordinates_mock_callback,
        event_info_callback=event_info_mock_callback,
        travel_time_callback=get_travel_time)


@pytest.fixture(params=list(DBS.values()))
def all_clients_all_callbacks(request):
    """
    Fixture returning all clients with all callbacks.
    """
    return create_async_client(
        request.param,
        station_coordinates_callback=station_coordinates_mock_callback,
        event_info_callback=event_info_mock_callback,
        travel_time_callback=get_travel_time)


def _add_callback(client):
    def request_callback(request):
        req = client.fetch(request.path_url)
        return (req.code, req.headers, req.body)

    pattern = re.compile(r"http://localhost.*")
    responses.add_callback(
        responses.GET, pattern,
        callback=request_callback,
        content_type="application/octet_stream"
    )


@pytest.fixture(params=list(DBS.values()))
@responses.activate
def all_remote_dbs(request):
    client = create_async_client(request.param, None)

    _add_callback(client)

    db = instaseis.open_db("http://localhost:%i" % client.port)
    db._client = client
    return db

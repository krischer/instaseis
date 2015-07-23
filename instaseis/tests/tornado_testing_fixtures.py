#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The tornado.testing Async stuff packaged as fixtures for use with py.test

Originally from https://gist.github.com/robcowie/7843633; modified for the
instaseis server.
"""
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

import instaseis
from instaseis.server.app import application
from instaseis.instaseis_db import InstaseisDB


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


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

    def fetch(self, path, **kwargs):
        # This seems a bit fragile. How else to get the dynamic port number?
        port = list(self.httpserver._sockets.values())[0].getsockname()[1]
        url = u'%s://localhost:%s%s' % ('http', port, path)
        self.httpclient.fetch(url, self.stop, **kwargs)
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

    def __rethrow(self):
        if self.__failure is not None:
            failure = self.__failure
            self.__failure = None
            raise_exc_info(failure)

    def wait(self, condition=None, timeout=None):
        if timeout is None:
            timeout = 5

        if not self.__stopped:
            if timeout:
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
    [sock] = netutil.bind_sockets(None, 'localhost', family=socket.AF_INET)
    port = sock.getsockname()[1]
    return sock, port


# All test databases.
DBS = {
    "db_bwd_displ_only": os.path.join(DATA, "100s_db_bwd_displ_only"),
    "db_bwd_strain_only": os.path.join(DATA, "100s_db_bwd_strain_only"),
    "db_fwd": os.path.join(DATA, "100s_db_fwd"),
    "db_fwd_deep": os.path.join(DATA, "100s_db_fwd_deep")
}


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
    else:
        return []


def create_async_client(path, station_coordinates_callback):
    application.db = InstaseisDB(path)
    application.station_coordinates_callback = station_coordinates_callback
    # Build server as in testing:311
    sock, port = bind_unused_port()
    server = HTTPServer(application, io_loop=IOLoop.instance())
    server.add_sockets([sock])
    client = AsyncClient(server, AsyncHTTPClient())
    client.filepath = path
    client.port = port
    return client


@pytest.fixture(params=list(DBS.values()))
def all_clients(request):
    """
    Fixture returning all clients!
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

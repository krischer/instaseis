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
import socket
import sys

from tornado import netutil
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.testing import AsyncHTTPClient
from tornado.util import raise_exc_info

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
        ## We're using a singleton ioloop throughout
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


@pytest.fixture()
def client_db_bwd_displ_only(request):
    application.db = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

    ## Build server as in testing:311
    sock, port = bind_unused_port()
    server = HTTPServer(application, io_loop=IOLoop.instance())
    server.add_sockets([sock])

    return AsyncClient(server, AsyncHTTPClient())
